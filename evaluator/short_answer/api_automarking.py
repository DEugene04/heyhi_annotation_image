from pydantic import BaseModel, field_validator

from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request, HTTPException
from tools import utils

from typing import List, Dict, Union, Literal

import os, json
import re
from bs4 import BeautifulSoup

from openai import OpenAI, AsyncOpenAI
import time
import asyncio

from decimal import Decimal
from typing import Any

from evaluator.short_answer.chain import AutoMarkingChain
from evaluator.short_answer.logic import QuestionExtractor, RubricTransformer
from evaluator.short_answer.logic import AutoMarking, MarkAllocation, GenerateKBAnswer, QuestionContext, AnswerContext, RubricContext, QuestionMarkContext, AutomarkingUserPromptConstructor, MarkingResultContext

import config

## Set up logger
import logging
logger = logging.getLogger("logger")

## Pydantic Models

## Define API Router
router = APIRouter(
    prefix="/automarking",
    tags=['AutoMarking System API']
)

## API Endpoints
@router.post("/automarking-predict")
async def auto_marking(request: Request, reqs: AutoMarking):
    """
    An API endpoint to automark student's answer based on given question & model answer.
    To use the endpoint, use the following JSON body:
    ```
    {
        "question": A JSON string with the following format:
            '''
            [
                {'type': 'text', 'text': ''},
                {'type': 'image_url', 'image_url': {'url': ''}}
            ]
            ''',
        "student_answer": student's answer to the following model,
        "full_mark: the full mark of the answer. 1 means the score ranges from 0 to 1.
        "step": the mark increment of the score from lower to upper boundary. The default value is 0.5.
        "fitb_index": only for FITB questions. it indicates the index number of which blank to fill. is optional.
        "rubric": A JSON string with the following format:
            '''
            [
                {
                    "full_mark": "",
                    "half_mark": "",
                    "zero_mark": "",
                }
            ]
            '''
        "answer_pas": a string of mark-allocated correct answer. The default value is "",
        "incorrect_flags": a list of strings indicating the incorrect flags for the answer. If the student answer contains any of the incorrect flags, the answer will be immediately be marked as incorrect. The default value is [],
        "marking_note": additional note for marking. Used to assist the AI further for the specific question. The default value is "",
        "subject": subject of the question. Used to assist the AI further for the specific question. The default value is "",
    }
    ```
    """
    
    ## Get QuestionContext instance, to construct the question, answer, and marking context that will be used for ALL the marking process
    total_tokens = 0
    total_cost = 0
    question_extractor = QuestionExtractor(reqs = reqs)
    question_context = await question_extractor.constructQuestionContext(request.state.openai_api_key)
    models = set()

    print(question_context.to_dict())

    retries = 0
    max_retries = 3
    
    while retries < max_retries:
        try:
            print(f"Attempt number: {retries + 1}")

            # Get AI Marking Result
            result, result_total_tokens, result_total_cost, models_automarking = await AutoMarkingChain.get_result(
                question_context = question_context, 
                reqs = reqs, 
                debug_mode = reqs.debug_mode,
                auto_fail = reqs.auto_fail,
                api_key = request.state.openai_api_key
            )

            total_tokens += result_total_tokens
            total_cost += result_total_cost
            models.update(models_automarking)

            # Get AI Marking Highlight Breakdown
            marking_result_context = MarkingResultContext(
                ai_mark = result['mark'],
                reason = result['reason']
            )

            result_highlights, breakdown_total_tokens, breakdown_total_cost, models_breakdown = await AutoMarkingChain.get_automarking_breakdown(
                question_context = question_context, 
                marking_result_context = marking_result_context, 
                debug_mode = reqs.debug_mode,
                auto_fail = reqs.auto_fail,
                api_key = request.state.openai_api_key
            )

            total_tokens += breakdown_total_tokens
            total_cost += breakdown_total_cost
            models.add(models_breakdown)

            # Construct Full Result
            type_fitb = question_context.mark_context.type_fitb
            fitb_index = question_context.mark_context.fitb_index
            unique_blanks = question_context.mark_context.unique_blanks

            if type_fitb:
                if isinstance(fitb_index, list):
                    remarks = result_highlights['remarks_simplified']
                    remarks_detailed = result_highlights['remarks']
                else:
                    remarks = f"For blank {fitb_index}: {result_highlights['remarks_simplified']}" if type_fitb and unique_blanks > 1 else result_highlights['remarks_simplified']
                    remarks_detailed = f"For blank {fitb_index}: {result_highlights['remarks']}" if type_fitb and unique_blanks > 1 else result_highlights['remarks']
            else:
                remarks = result_highlights['remarks_simplified']
                remarks_detailed = result_highlights['remarks']

            full_result = {
                # 'student_answer': result['student_answer'] if not isinstance(fitb_index, list) else question_context.answer_context.student_answer[0],
                'student_answer': question_context.answer_context.student_answer if not isinstance(fitb_index, list) else question_context.answer_context.student_answer[0],
                'mark': result['mark'],
                'reason': result['reason'],
                'remarks': remarks,
                'remarks_detailed': remarks_detailed,
                'mark_highlights': result_highlights['mark_highlights'],
                'messages_automarking': result.get('messages_automarking', []),
                'messages_breakdown': result_highlights.get('messages_breakdown', []),
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'models_used': list(models)
            }

            correct_answer = question_context.answer_context.original_correct_answer if type_fitb else question_context.answer_context.correct_answer

            if isinstance(correct_answer, list):
                try:
                    correct_answer = correct_answer[0]
                except IndexError:
                    correct_answer = ""

            try:
                student_answer = reqs.student_answer
            except Exception as e:
                student_answer = ""
                        
            return JSONResponse(utils.add_project_info(
                    {
                        "result": full_result,
                        "correct_answer": correct_answer,
                        "student_answer": student_answer
                    }, 2, total_tokens, total_cost, list(models), 0
                )
            )
        
        except Exception as e:
            print(f"Trial number {retries + 1} out of {max_retries + 1}...")
            print(f"Error details: {e}")
            retries += 1
            if retries < max_retries:
                time.sleep(3)
            else:
                raise Exception(f"OpenAI API failed after multiple attempts.\nError details: {e}") from e

@router.post("/generate_mark_allocation")
async def generate_mark_allocation(request: Request, reqs: MarkAllocation):
    client = AsyncOpenAI(api_key = request.state.openai_api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
    question_extrator = QuestionExtractor(reqs = reqs)
    question_context = await question_extrator.constructQuestionContext(request.state.openai_api_key)
    vectorstore = False

    system_prompt = """
        You are an assistant that is responsible for allocating marks on a model/correct answer for a question.

        Since the questions you will be dealing with are Open Ended Questions, the maximum mark for the answer can vary.

        Your task is to look into the maximum mark you will be given for the question, as well as the mark increment/step.

        A question with maximum mark of 1 will have increment of 0.5, and will have two possible marks: 0, 0.5, 1.
        Question with mark more than 1 (for example, 2, 3, or even 4). The increment will start from 1, and not 0.5 anymore. So for example, a maximum mark of 3 with increment of 1 would have 4 possible marks: 0, 1, 2, and 3.

        After that, you will be given the correct answer for the question, and your task is to transform the correct answer by adding marks to each sentence/information inside the correct answer. The added mark should be the same as the value of the increment/step. For example, a full mark of 2 would produce two (1) labels at most, since the increment can only start from 1 and not from 0.5 anymore.

        For example, you would want to transform "He was afraid he might drown, be eaten by predatory fish, or miss attracting the attention of rescuers." into "He was afraid he might drown (1), be eaten by predatory fish (1), or miss attracting the attention of rescuers (1).", if the maximum mark is 3.

        However, for cases where the answers are separated within the / mark, treat them as alternative answers. Which means, each alternative answer would get the full mark, instead of dividing them into smaller marks. For example, if the maximum mark is 1 and the correct answer is "He was afraid he might drown / be eaten by predatory fish / miss attracting the attention of rescuers.", it means there are three acceptable correct answers and each of them would get 1 full mark, and the label would be "He was afraid he might drown / be eaten by predatory fish / miss attracting the attention of rescuers. (1)". For this case, do not label each alternative answer, and only put the label at the end of the sentence.

        BUT, if the usage of / is to indicate synonym/similar words (for example: "he is sad/stressed/exhausted because of the things he felt"), then we can allow partial marks to come and marks can be added to each information insdie the correct answer. This is only IF the usage of / is to indicate synonyms/similar words.

        If there are BOTH usage of / (one to indicate synonyms, and one to indicate alternative answers), we can also still allow for partial marks to come and marks can still be added to each information inside the correct answer, BUT DO NOT GIVE MARKS TO EACH SYNONYM. For example, if the answer is "he is sad/stressed/exhausted because of the things he felt / he was not ready for the experience" and the mark is 2, then we can label them into "he is sad/stressed/exhausted because of the things he felt (1) / he was not ready for the experience (1)".

        This way, the correct answer will now clearly describe the importance of completeness of an answer, which will be used further as the baseline when marking a student's answer.

        You should also return the detailed version of the correct answer. This version splits each individual marked phrase/sentence. Only split according to the number of labels generated. So for answers with alternative answers, since the label is only one at the end of the sentence, you don't need to split it.

        **Note:** if the given correct answer already has mark allocated to it, then you can just ignore by returning the same correct answer without transforming it.
    """

    system_prompt += """
        Return your response in the following JSON format:
        {
            "previous_answer": previous correct answer before transforming process,
            "current_answer": current correct answer, after performing the transformation process based on the full mark and increment given,
            "detail": [
                {
                    "sentence": sentence/phrase 1,
                    "mark": the allocated mark for the sentence/phrase
                },
                {
                    "sentence": sentence/phrase 2,
                    "mark": the allocated mark for the sentence/phrase
                },
                ...
            ]
        }
    """
    
    system_prompt_object = {
        "role": "system", 
        "content": [
            {
                "type": "text", 
                "text": system_prompt
            }
        ]
    }

    messages = []
    user_prompt_image = []

    question_obj = await AutoMarkingChain.get_user_prompt(f"Here is the composition & question: {question_context.question}", "text", vectorstore)
    model_answer_obj = await AutoMarkingChain.get_user_prompt(f"Here is the correct answer you need to transform: `{question_context.answer_context.correct_answer}`. The full mark is {str(question_context.mark_context.full_mark)} with increment of can be 0.5 or 1, the most important thing is the total of mark allocation should be equal to the full mark.", "text", vectorstore)

    for item in question_context.string_urls:
        user_image_obj = await AutoMarkingChain.get_user_prompt(item, "image_url", vectorstore)
        user_prompt_image.append(user_image_obj)

    messages.append(system_prompt_object)
    messages.append(question_obj)
    if user_prompt_image:
        for prompt_image in user_prompt_image:
            messages.append(prompt_image)
    messages.append(model_answer_obj)

    retries = 0
    max_retries = 10

    while retries < max_retries:
        try:
            print(f"Attempt number: {retries + 1}")
            response = await client.chat.completions.create(
                model = "gpt-4o",
                messages = messages,
                # temperature = 0,
                # max_tokens = 4095,
                # top_p = 1,
                response_format = {
                    "type": "json_object"
                }
            )

            result = json.loads(response.choices[0].message.content)
            total_tokens = response.usage.total_tokens
            total_cost = (response.usage.prompt_tokens * 2.5 + response.usage.completion_tokens * 10)/1e6
            return JSONResponse(utils.add_project_info(result,2,total_tokens,total_cost,["gpt-4o"], 0))
        
        except Exception as e:
            print(f"Trial number {retries + 1} out of {max_retries + 1}...")
            retries += 1
            if retries < max_retries:
                time.sleep(3)
            else:
                raise Exception(f"OpenAI API failed after multiple attempts.\nError details: {e}") from e

@router.post("/generate-kb-answers")
async def generate_kb_answer(request: Request, reqs: GenerateKBAnswer):
    total_tokens = 0
    total_cost = 0

    ## loading the data
    question = reqs.question
    file_id = reqs.file_id
    classification = reqs.classification
    
    result, result_total_tokens, result_total_cost = await AutoMarkingChain.get_automarking_answers(question, file_id, classification, request.state.openai_api_key)
    total_tokens += result_total_tokens
    total_cost += result_total_cost

    return JSONResponse(utils.add_project_info(
        {
            "possible_answers": result['possible_answers'],
        }, 2, total_tokens, total_cost, ["gpt-4o", "gpt-4o-mini"], 0
    ))