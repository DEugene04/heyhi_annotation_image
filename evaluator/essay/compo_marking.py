from openai import OpenAI, AsyncOpenAI
import json
import config
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from pydantic import BaseModel, Field
from evaluator.essay.chain import EssayMarkingChain
from evaluator.dynamic_llm_call import DynamicAsyncOpenAI

class MarkingResult(BaseModel):
    feedback: str = Field(description="Your feedback in one paragraph")
    mark: float = Field(description="Your mark")

class CombinedMarkingResultSingle(BaseModel):
    id: int
    name: str
    feedback: str
    mark: float

class CombinedMarkingResult(BaseModel):
    all_criteria: list[CombinedMarkingResultSingle]

class Paragraph(BaseModel):
    lines: list[str]

class ImageResult(BaseModel):
    image_id:int
    handwritten_text:str
    paragraphs: list[Paragraph]


async def compo_marking_dependent_rubric_switching(question_statement, rubric_table, student_composition, model_composition, student_class, language, examples = [], total_tokens = 0, total_cost = 0, api_key = None):
    ## check rubric interdependency
    result, usage, model = await EssayMarkingChain.check_rubric_interdependency(rubric_table, student_class, api_key)
    
    ## GPT-4o
    total_cost += usage.total_cost
    total_tokens += usage.total_tokens
    models = set([model])

    ## run the compo marking chain
    try:
        if result['is_dependent'] == True:
            print(f"Rubric is dependent, switching to combined marking approach.")
            return await EssayMarkingChain.compo_marking_v3_combined(
                question_statement = question_statement, 
                rubric_table = rubric_table, 
                student_composition = student_composition,
                model_composition = model_composition, 
                student_class = student_class, 
                language = language, 
                examples = examples, 
                total_tokens = total_tokens, 
                total_cost = total_cost,
                models = models,
                api_key = api_key
            )
        else:
            print(f"Rubric is not dependent, switching to separate marking approach.")
            return await EssayMarkingChain.compo_marking_v3(
                question_statement = question_statement,
                rubric_table = rubric_table,
                student_composition = student_composition,
                model_composition = model_composition,
                student_class = student_class,
                language = language,
                examples = examples,
                total_tokens = total_tokens,
                total_cost = total_cost,
                models = models,
                api_key = api_key
            )
    except Exception as e:
        print(f"Error in 'compo_marking_dependent_rubric_switching': {e}")
        return await EssayMarkingChain.compo_marking_v3(
            question_statement = question_statement,
            rubric_table = rubric_table,
            student_composition = student_composition,
            model_composition = model_composition,
            student_class = student_class,
            language = language,
            examples = examples,
            total_tokens = total_tokens,
            total_cost = total_cost,
            models = models,
            api_key = api_key
        )

async def generate_good_points(question_statement, rubric_table, marking_result, student_essay, api_key):
    client = DynamicAsyncOpenAI(api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
    model = "gpt-4.1-mini"
    system_prompt = """Evaluate a student's essay according to the provided question statement and the marking of the essay by identifying and explaining the strengths of the essay, focusing solely on Higher-Order Concerns (HOCs) like ideas, evidence, organization, etc. Do not comment on Lower-Order Concerns (LOCs) like grammar, punctuation, or word choice. For each identified positive HOC, cite a specific sentence or short excerpt from the student's essay as concrete evidence for your evaluation. Ensure thorough feedback, explaining why each highlighted aspect meets the criteria before making your assessment. The feedback should refer to the student as "you" to act as a detailed comment for the student.

Think step-by-step: First, analyze the essay in relation to HOCs, outlining the HOCs where the student meets or exceeds expectations. Then, choose representative sentences or a short excerpt from the essay as evidence for each positive point. After selecting the excerpt, summarize your overall positive feedback. Continue until all positive HOC items are identified and properly evidenced.

## Detailed Steps
- Analyze the student's essay, focusing on rubric criteria related to Higher-Order Concerns only.
- For each positive HOC, explain your reasoning: specify which rubric criterion is met, why it is important, and how the essay demonstrates this.
- Select a representative sentence or short excerpt from the essay that exemplifies each positive concern.
- Summarize your positive feedback for the excerpt chosen.
- Exclude comments on grammar, spelling, sentence-level mechanics, or word choice.

**Important:**  
- Focus exclusively on Higher-Order Concerns.  
- Present evidence before giving positive feedback. 
- Excerpt cannot have "...", the excerpt must be a direct quote from the essay.
- Excerpts must be unique from one another.
- The essay may include the question, instructions, title, or numbering transcribed from the page. That text is NOT the student's writing: do not comment on it and never use it as an excerpt.
- Use the specified JSON format.  
- Repeat step-by-step analysis for each positive HOC observed."""

    question_message = "Question statement:\n{question_statement}".format(question_statement=question_statement)
    
    formatted_marking_results = []
    for rubric in rubric_table:
        marking_result_rubric = next(filter(lambda x: x["component"] == rubric["name"], marking_result["score_detail"]), None)
        if marking_result_rubric == None:
            rubric_possible_scores = [criterion["score"] for criterion in rubric["breakdown"]] if rubric["score_type"] == "fixed" else [score for criterion in rubric["breakdown"] for score in [criterion["from_score"], criterion["to_score"]]]
            formatted_marking_results.append(
                {
                    "name": rubric["name"],
                    "max_score": rubric["max_score"],
                    "score": min(rubric_possible_scores),
                    "feedback": "(rubric is missing marking)"
                }
            )
        else:
            formatted_marking_results.append(
                {
                    "name": rubric["name"],
                    "max_score": rubric["max_score"],
                    "score": marking_result_rubric["score"],
                    "feedback": marking_result_rubric["feedback"]
                }
            )
    
    formatted_marking_message = "\n".join(
        [
            "- {rubric_result}".format(rubric_result=rubric_result) for rubric_result in
            [
                "Rubric: {rubric}\nMark: {score}/{max_score}\nFeedback: {feedback}".format(
                    rubric = formatted_marking_result["name"],
                    score = formatted_marking_result["score"],
                    max_score = formatted_marking_result["max_score"],
                    feedback = formatted_marking_result["feedback"]
                ) for formatted_marking_result in formatted_marking_results
            ]
        ]
    )

    student_essay_message = "Student's essay:\n{student_essay}".format(student_essay=student_essay)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question_message},
        {"role": "user", "content": formatted_marking_message},
        {"role": "user", "content": student_essay_message}
    ]

    json_schema = {
        "name": "positive_point_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "positive_points": {
                    "type": "array",
                    "description": "List of positive point with evidence and positive feedback.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "excerpt": {
                                "type": "string",
                                "description": "The excerpt for this point."
                            },
                            "feedback": {
                                "type": "string",
                                "description": "Positive feedback on why the student meets the expectation or exceeds it for this point."
                            }
                        },
                        "required": [
                            "excerpt",
                            "feedback"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": [
                "positive_points"
            ],
            "additionalProperties": False
        }
    }

    response = await client.responses.create(
        model = model,
        input = messages,
        text = {"format": {"type": "json_schema", **json_schema}},
        temperature = 0,
        top_p = 1
    )

    total_tokens = response.usage.total_tokens
    total_cost = response.usage.total_cost

    return {"response": json.loads(response.output_text)["positive_points"], "total_tokens": total_tokens, "total_cost": total_cost, "models": {response.model}}