import time
from typing import List

from pydantic import BaseModel
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from evaluator.essay.compo_marking import compo_marking_dependent_rubric_switching, generate_good_points
from tools import utils

class CompoMarkingRequest(BaseModel):
    question_statement: str = ""
    rubric_table: List[dict]
    student_composition: str = ""
    model_composition: str
    student_class: str
    question_annotation: str = ""
    language: str = "British English"
    is_flc: bool = False
    highlight_good_points: bool = False
    examples: List[dict] = []

router = APIRouter(
    prefix="/compo-marking",
    tags=['Compo Marking/Rubric Marking System API']
)

URL_QUERY = "https://ahkmhmeahoyfrd3vaxzfok26ny0ybyfr.lambda-url.ap-southeast-1.on.aws/dba/query_update"

@router.post("/get-score-v3")
async def compo_marking_api_v3(request: Request, req: CompoMarkingRequest):
    start_time = time.perf_counter()
    total_tokens = 0
    total_cost = 0
    models = set()

    # Reject any request that has invalid rubric table
    for rubric in req.rubric_table:
        if rubric["score_type"] == "fixed":
            if any(not isinstance(item["score"], (int, float)) for item in rubric["breakdown"]):
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": f"Invalid score in rubric table for the rubric {rubric['name']} "
                                f"(required to be of type integer or float)"
                    },
                )

        elif rubric["score_type"] == "range":
            if any(not isinstance(item["from_score"], (int, float)) for item in rubric["breakdown"]):
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": f"Invalid from_score in rubric table for the rubric {rubric['name']} "
                                f"(required to be of type integer or float)"
                    },
                )

            elif any(not isinstance(item["to_score"], (int, float)) for item in rubric["breakdown"]):
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": f"Invalid to_score in rubric table for the rubric {rubric['name']} "
                                f"(required to be of type integer or float)"
                    },
                )
    
    # Rubric marking
    compo_marking_response = await compo_marking_dependent_rubric_switching(
        question_statement = req.question_statement, 
        rubric_table = req.rubric_table, 
        student_composition = req.student_composition, 
        model_composition = req.model_composition, 
        student_class = req.student_class, 
        language = req.language, 
        examples = req.examples,
        api_key = request.state.openai_api_key
    )
    total_tokens += compo_marking_response["total_tokens"]
    total_cost += compo_marking_response["total_cost"]
    models.update(compo_marking_response["models"])

    response = {
        "feedback": compo_marking_response["feedback"]
    }

    # "Good points" highlighting
    if req.highlight_good_points == True:
        good_point_response = await generate_good_points(req.question_statement, req.rubric_table, compo_marking_response["feedback"], req.student_composition, request.state.openai_api_key)
        total_tokens += good_point_response["total_tokens"]
        total_cost += good_point_response["total_cost"]
        models.update(good_point_response["models"])

        modified_good_point_response = [
            {
                "target": criterion["excerpt"],
                "comment": criterion["feedback"]
            } for criterion in good_point_response["response"] if criterion["excerpt"] in req.student_composition
        ]
        response["good_points"] = modified_good_point_response

    end_time = time.perf_counter()
    response["duration"] = end_time - start_time

    return utils.add_project_info(response, 8, total_tokens, total_cost, list(models), 0)