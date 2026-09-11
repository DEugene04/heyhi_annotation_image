from openai import OpenAI, AsyncOpenAI
from bs4 import BeautifulSoup
import json
import time
import re
import asyncio
import math

from pydantic import BaseModel, Field

from tools import utils
from evaluator.essay import image_source
from evaluator.essay.logic import EssayMarkingPromptConstructor
from evaluator.dynamic_llm_call import DynamicAsyncOpenAI

import config

class MarkingResult(BaseModel):
    feedback: str = Field(description="Your feedback in one paragraph")
    feedback_simplified: str = Field(description="Your feedback in one paragraph but simplified for students to understand")
    is_matching_with_example: bool = Field(description="Check if the essay provided matches any of the examples provided")
    mark: float = Field(description="Your mark")

class CombinedMarkingResultSingle(BaseModel):
    id: int
    name: str
    feedback_in_depth: str
    feedback: str
    mark: float

class CombinedMarkingResult(BaseModel):
    all_criteria: list[CombinedMarkingResultSingle]

async def get_completion_turbo(prompt, model = "gpt-3.5-turbo", question = "", max_tokens = 4000, api_key = config.OPENAI_API_KEY_DICT["GLOBAL"]):
    messages = [
        {"role": "system", "content": prompt}, 
        {"role": "user", "content": question}
    ]
    
    client = DynamicAsyncOpenAI(api_key = api_key)
    response = await client.chat.completions.create(
        model = model,
        messages = messages,
        max_tokens = max_tokens,
        temperature = 0,
        response_format = {"type": "json_object"}
    )

    return response.output_text, response.usage, response.model

def criteria_breakdown_range_flatten(criteria):
    component_id = None
    full_breakdown = ""
    breakdown = ""
    if criteria['score_type'] == "range":
        if len(criteria['breakdown']) > 1 and criteria['breakdown'][0]['from_score'] > criteria['breakdown'][1]['from_score']:
            for i in range(len(criteria['breakdown'])):
                if criteria['breakdown'][i].get('component_id', None) != component_id and criteria['breakdown'][i].get('component_name', None):
                    full_breakdown += "Component: {component_name}\nRubric breakdown:\n{breakdown}\n\n".format(
                        component_name = criteria['breakdown'][i].get('component_name', None),
                        breakdown = breakdown
                    )
                breakdown += "{from_score}-{to_score}: {description}\n".format(
                    from_score = criteria['breakdown'][i]['from_score'],
                    to_score = criteria['breakdown'][i]['to_score'],
                    description = criteria['breakdown'][i]['description'],
                )
        else:
            for i in range(len(criteria['breakdown'])):
                if criteria['breakdown'][i].get('component_id', None) != component_id and criteria['breakdown'][i].get('component_name', None):
                    full_breakdown += "Component: {component_name}\nRubric breakdown:\n{breakdown}\n\n".format(
                        component_name = criteria['breakdown'][i].get('component_name', None),
                        breakdown = breakdown
                    )
                breakdown += "{from_score}-{to_score}: {description}\n".format(
                    from_score = criteria['breakdown'][i]['from_score'],
                    to_score = criteria['breakdown'][i]['to_score'],
                    description = criteria['breakdown'][i]['description']
                )
            
    elif criteria['score_type'] == "fixed":
        for i in range(len(criteria['breakdown'])):
            if criteria['breakdown'][i].get('component_id', None) != component_id and criteria['breakdown'][i].get('component_name', None):
                full_breakdown += "Component: {component_name}\nRubric breakdown:\n{breakdown}\n\n".format(
                    component_name = criteria['breakdown'][i].get('component_name', None),
                    breakdown = breakdown
                )
            breakdown += "Score {score}: {description}\n".format(
                score = criteria['breakdown'][i]['score'],
                description = criteria['breakdown'][i]['description']
            )

    if component_id != None:
        full_breakdown += "Component: {component_name}\nRubric breakdown:\n{breakdown}".format(
            component_name = criteria['breakdown'][-1].get('component_name', None),
            breakdown = breakdown
        )
    else:
        full_breakdown = breakdown

    return full_breakdown

def criteria_breakdown(criteria):
    possible_scores = set()
    if criteria['score_type'] == "range":
        # print(f"Criteria {criteria['name']} has range breakdown, flattening the breakdown for easier understanding of the possible scores...")
        breakdown = ""
        for i in range(len(criteria['breakdown'])):
            breakdown += "{from_score}-{to_score}: {description}\n".format(
                from_score = criteria['breakdown'][i]['from_score'],
                to_score = criteria['breakdown'][i]['to_score'],
                description = criteria['breakdown'][i]['description']
            )
            possible_scores.update([criteria['breakdown'][i]['from_score'], criteria['breakdown'][i]['to_score']])
    elif criteria['score_type'] == "fixed":
        # print(f"Criteria {criteria['name']} has fixed breakdown, flattening the breakdown for easier understanding of the possible scores...")
        breakdown = ""
        for i in range(len(criteria['breakdown'])):
            breakdown += "Score {score}: {description}\n".format(
                score = criteria['breakdown'][i]['score'],
                description = criteria['breakdown'][i]['description']
            )
            possible_scores.add(criteria['breakdown'][i]['score'])
    # print(f"Finished flattening the breakdown for criteria {criteria['name']}. Breakdown:\n{breakdown}")
    return breakdown, possible_scores

async def get_score_detail(criteria, score, feedback, feedback_detailed = ""):
    score_detail = None
    if criteria['score_type'] == "range":
        for breakdown in criteria['breakdown']:
            if breakdown["from_score"] <= score <= breakdown["to_score"]:
                score_detail = {
                    "component": criteria['name'],
                    "score": score,
                    "feedback": feedback,
                    "feedback_detailed": feedback_detailed,
                    "description": breakdown["description"],
                    "id": breakdown["id"]
                }
                break

    elif criteria['score_type'] == "fixed":
        for idx in range(len(criteria['breakdown'])):
            breakdown_current = criteria['breakdown'][idx]
            breakdown_next = criteria['breakdown'][idx + 1] if idx < len(criteria['breakdown']) - 1 else criteria['breakdown'][idx]
            if breakdown_current["score"] <= score <= breakdown_next["score"] or breakdown_next["score"] <= score <= breakdown_current["score"]:
                selected_breakdown = breakdown_current if abs(breakdown_current["score"] - score) < abs(score - breakdown_next["score"]) else breakdown_next
                score_detail = {
                    "component": criteria['name'],
                    "score": selected_breakdown["score"],
                    "feedback": feedback,
                    "feedback_detailed": feedback_detailed,
                    "description": selected_breakdown["description"],
                    "id": selected_breakdown["id"]
                }

                # print(f"Score detail for {criteria['name']}: {score_detail}")
                break
    return score_detail

async def construct_essay_marking_feedback_object(
    essay_marking_result,
    rubric_table,
    new_rubric_table,
):
    feedback_list = []
    score_detail = []
    final_score = 0
    
    for result_student in essay_marking_result["all_criteria"]:
        criteria = next(filter(lambda x: x['id'] == result_student['id'], rubric_table), rubric_table[0])
        new_criteria = next(filter(lambda x: x['id'] == result_student['id'], new_rubric_table), new_rubric_table[0])
        feedback_list.append({
            "criteria_name": new_criteria['name'],
            "feedback": result_student['feedback'],
            "feedback_detailed": result_student['feedback_in_depth']
        })
        print(f"student feedback: {result_student['feedback_in_depth']}")
        print(f"Student's mark for {new_criteria['name']}: {result_student['mark']}")

        breakdown, possible_scores = criteria_breakdown(new_criteria)
        score = max(min(possible_scores), min(max(possible_scores), result_student['mark']))
        final_score += score

        ## Getting the description of each category score
        score_detail.append(await get_score_detail(
            criteria = criteria,
            score = score,
            feedback = result_student['feedback'],
            feedback_detailed = result_student['feedback_in_depth']
        ))

    feedback_string = ""
    feedback_string_detailed = ""
    for feedback in feedback_list:
        feedback_string += f"{feedback['criteria_name']}: {feedback['feedback']}\n"
        feedback_string_detailed += f"{feedback['criteria_name']}: {feedback['feedback_detailed']}\n"

    return {
        "final_score": final_score,
        "feedback_string": feedback_string,
        "feedback_string_detailed": feedback_string_detailed,
        "score_detail": score_detail
    }

class EssayMarkingPreprocessor:
    def __init__(self, model: str = "gpt-4o-mini", total_tokens = 0, total_cost = 0, api_key = None):
        self.client = DynamicAsyncOpenAI(api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
        self.model = model
        self.total_tokens = total_tokens
        self.total_cost = total_cost
        self.models = set()
    
    def _get_llm_data(self):
        return {
            'models': self.models,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost
        }

    async def get_image_description(self, image_url: str = "", question_statement: str = "") -> dict:
        client = self.client
        model = self.model
        
        image_description_structure = r"""
            {
                "objects": [
                    {
                        "name": ,
                        "description":
                    },
                    ...
                ],
                "description":

            }
        """

        system_prompt = """You will be given an image that is attached as a content or a solution for a question in a quiz. 
Your purpose is to give a clear description of the given image. To help you with this, you will also be given the question statement (if there are any).
Return your image description in the same language as the given question statement. If there are no question statement given, then return in English as the default.
Special case: If the image contains a bar chart of "Average Monthly Rent Apartments in 2020 (USD)", please use this information: New Yorks = $3,500, London = $2,000, Tokyo = $2,500, Sydney = $3,000, Berlin = $1,500
If the image contains a text, ensure to convert all the text to the description.
Your response should be in the following JSON format:
{image_description_structure}""".format(image_description_structure = image_description_structure)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        user_prompts = []

        if question_statement:
            question_statement_prompt = f"Here is the question statement:\n{question_statement}"
            user_prompts.append({"type": "text", "text": question_statement_prompt})
        
        user_prompts.append({"type": "image_url", "image_url": {"url": image_url}})

        messages.append({"role": "user", "content": user_prompts})

        ## get image description
        retries = 0
        max_retries = 5

        while retries < max_retries:
            try:
                response = await client.chat.completions.create(
                    model = model,
                    messages = messages,
                    max_tokens = 4000,
                    temperature = 0,
                )
                image_description = response.choices[0].message.content
                self.total_tokens += response.usage.total_tokens
                self.total_cost += response.usage.total_cost
                self.models.add(response.model)

                break
        
            except Exception as e:
                print(f"Trial number {retries + 1} out of {max_retries + 1}...")
                retries += 1
                if retries < max_retries:
                    time.sleep(3)
                else:
                    image_description = "No image description"
                    break

        ## parse image description
        img_desc = "-"
        if image_description != '-':
            try:
                pattern = r"\{(.|\n)*\}"
                match = re.search(pattern, image_description)
                if match:
                    img_desc = json.loads(match.group(0))
                img_desc = image_description
            
            except:
                img_desc = "No image description."
        
        return {
            "image_description": f"Image Description: '{img_desc}'"
        }
    
    async def _preprocess_question_statement(self, question_statement):
        img_urls = []
        try:
            question_statement = question_statement.replace("\\/", "/")
            soup = BeautifulSoup(question_statement, 'html.parser')

            for img in soup.find_all('img'): ## if image in question statement
                new_tag = soup.new_tag("p")
                img.insert_before(new_tag)

                # Resolve the src to a value the grader can consume directly: a
                # public URL passes through, while an s3:// reference or a local
                # path is fetched and inlined as a base64 data URI. This is the
                # single choke point, so prod (S3) and tests (local files) share
                # one code path -- only the reference string differs.
                try:
                    src = image_source.to_data_uri(img['src'])
                except image_source.ImageSourceError as e:
                    print(f"Skipping unresolvable question image {img.get('src')!r}: {e}")
                    img.decompose()
                    continue

                img_urls.append(src)

                # image_description_dict = await self.get_image_description(img['src'], question_statement)
                
                # new_tag.string = image_description_dict["image_description"]
                img.decompose()
            
            question_statement = str(soup.text)
        
        except Exception as e:
            print(f"Error in preprocessing question statement: {e}")

        return question_statement, img_urls
    
    async def get_word_count_llm(self, composition):
        client = self.client
        model = "gpt-5-mini"

        system_prompt = """Count the number of words in this text according to the text language, ignoring all HTML tags:
- If the language is Chinese, Korean, etc. that has characters as words, count the number of characters, but punctuation should not count.
- Else count the number of words."""
        
        json_schema = {
            "name": "word_count_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "word_count": {
                        "type": "integer"
                    }
                },
                "additionalProperties": False,
                "required": [
                    "word_count"
                ]
            }
        }

        response = await client.responses.create(
            model = model,
            input = composition,
            instructions = system_prompt,
            reasoning = {"effort": "low"},
            text = {"format": {"type": "json_schema", **json_schema}}
        )

        total_tokens = response.usage.total_tokens
        total_cost = response.usage.total_cost
        model = response.model

        return {
            "output": json.loads(response.output_text)["word_count"], 
            "total_tokens": total_tokens, 
            "total_cost": total_cost, 
            "model": model
        }
    
    async def _get_essay_word_count(self, student_composition: str = "", model_composition: str = ""):
        model_word_count = student_word_count = 0
        
        # Get composition word count
        student_word_count_response = await self.get_word_count_llm(student_composition)
        student_word_count = student_word_count_response["output"]
        self.total_cost += student_word_count_response["total_cost"]
        self.total_tokens += student_word_count_response["total_tokens"]
        self.models.add(student_word_count_response["model"])
        
        if model_composition:
            model_word_count_response = await self.get_word_count_llm(model_composition)
            model_word_count = model_word_count_response["output"]
            self.total_cost += model_word_count_response["total_cost"]
            self.total_tokens += model_word_count_response["total_tokens"]
            self.models.add(model_word_count_response["model"])
            
        return student_word_count, model_word_count

class EssayMarkingEvaluator:
    def __init__(self, model: str = "gpt-4o-mini", total_tokens = 0, total_cost = 0, api_key = None):
        self.client = DynamicAsyncOpenAI(timeout = 90, api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
        self.model = model
        self.total_tokens = total_tokens
        self.total_cost = total_cost
        self.models = set()
    
    def _get_llm_data(self):
        return {
            'models': self.models,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost
        }

    def _refine_rubric_table_prompt(self, criteria, highest_breakdown_score, highest_breakdown_id, model_solution):
        output_structure = r"""
{
    "description": <your description>
}"""

        prompt = """You are given a rubric table for marking a composition. The rubric table contains the following criteria: {criteria_name}. \
The highest score for {criteria_name} is {highest_breakdown_score}. \
The description for the highest score is as follows: {description}. \
Based on the model solution, which is as follows: \
```
{model_solution}
```
Refine the description for the highest score for {criteria_name} to better match the model solution. \
Your response should use the model solution as a reference to explain in more detail what is expected for the highest score. \
Your response should be in the following JSON format:
{output_structure}
""".format(
    criteria_name = criteria['name'], 
    highest_breakdown_score = highest_breakdown_score,
    description = criteria['breakdown'][highest_breakdown_id]['description'], 
    model_solution = model_solution,
    output_structure = output_structure
)
        return prompt

    async def refine_rubric_table_v2(self, rubric_table, model_solution):
        model = "gpt-4o-mini"

        for criteria in rubric_table:
            highest_breakdown_id = -1
            highest_breakdown_score = 0

            if criteria['score_type'] == "range":
                # Find the highest breakdown
                for i in range(len(criteria['breakdown'])):
                    if criteria['breakdown'][i]['to_score'] > highest_breakdown_score:
                        highest_breakdown_score = criteria['breakdown'][i]['to_score']
                        highest_breakdown_id = i

            elif criteria['score_type'] == "fixed":
                for i in range(len(criteria['breakdown'])):
                    if criteria['breakdown'][i]['score'] > highest_breakdown_score:
                        highest_breakdown_score = criteria['breakdown'][i]['score']
                        highest_breakdown_id = i

            # Refine the description of the highest breakdown
            prompt = self._refine_rubric_table_prompt(
                criteria, highest_breakdown_score, highest_breakdown_id, model_solution
            )

            response, usage, model = await get_completion_turbo(prompt, model, api_key = config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
            criteria['breakdown'][highest_breakdown_id]['description'] = json.loads(response)['description']
            
            # GPT-4o mini
            self.total_cost += utils.calculate_openai_usage_cost(
                model_name = model,
                input_tokens = usage.prompt_tokens,
                output_tokens = usage.completion_tokens
            )
            self.total_tokens += usage.total_tokens

        return rubric_table

    async def criteria_marking_new_combined(self, question_statement, rubric_components, student_composition, word_count, student_class, language, examples = [], img_urls = [], is_ielts = False):
        model = "gpt-5.4-mini"
        
        constructor = EssayMarkingPromptConstructor(
            is_dependent = True,
            question_statement = question_statement,
            student_composition = student_composition,
            rubric_components = rubric_components,
            student_class = student_class,
            language = language,
            word_count = word_count,
            examples = examples,
            img_urls = img_urls,
            is_ielts = is_ielts
        )

        messages = await constructor._construct_messages_object()
        max_iter = 3

        # print(f"Messages:\n=================================\n{messages}\n==================================")

        for ii in range(max_iter):
            try:
                response = await self.client.responses.parse(
                    model = model,
                    input = messages,
                    # temperature = 0,
                    # max_output_tokens = 4095,
                    # top_p = 1,
                    reasoning = {"effort": "none"},
                    text_format = CombinedMarkingResult
                )

                result = json.loads(response.output_text)
                self.total_tokens += response.usage.total_tokens
                self.total_cost += response.usage.total_cost
                self.models.add(response.model)

                return {
                    "output": result
                }
                    
            except Exception as e:
                print(f"Error in 'criteria_marking_new_combined': {e}")
                if ii < max_iter - 1:
                    pass
                else:
                    raise Exception("Rubric marking max iteration reached.")

    async def criteria_marking_new(self, question_statement, criteria, student_composition, word_count, student_class, breakdown, language, rubric_examples = [], img_urls = [], is_ielts = False):
        # model = "gpt-4.1-mini"
        model = "gpt-5-mini"

        constructor = EssayMarkingPromptConstructor(
            is_dependent = False,
            question_statement = question_statement,
            student_composition = student_composition,
            component = criteria,
            student_class = student_class,
            language = language,
            word_count = word_count,
            examples = rubric_examples,
            img_urls = img_urls,
            rubric_breakdown = breakdown,
            is_ielts = is_ielts
        )

        messages = await constructor._construct_messages_object()

        # print(f"Messages:\n=================================\n{messages}\n==================================")

        max_iter = 3

        for ii in range(max_iter):
            try:
                response = await self.client.responses.parse(
                    model = model,
                    input = messages,
                    # temperature = 0,
                    # max_output_tokens = 4095,
                    # top_p = 1,
                    reasoning = {"effort": "low"},
                    text_format = MarkingResult
                )

                result = json.loads(response.output_text)
                self.total_tokens += response.usage.total_tokens
                self.total_cost += response.usage.total_cost
                self.models.add(response.model)

                return {
                    "output": result
                }
                    
            except Exception as e:
                print(f"Error in 'criteria_marking_new': {e}")
                if ii < max_iter - 1:
                    pass
                else:
                    raise Exception("Rubric marking max iteration reached.")
        
        ## uncomment if need to change back to gemini
        # http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3))
        # with genai.Client(vertexai=False, api_key=config.GEMINI_API_KEY, http_options = http_options) as client:
        #     response = await client.aio.models.generate_content(
        #         model = "gemini-3-flash-preview",
        #         contents = input_prompt,
        #         config = types.GenerateContentConfig(
        #             system_instruction=system_prompt,
        #             response_mime_type="application/json",
        #             response_schema=MarkingResult,
        #             thinking_config=types.ThinkingConfig(thinking_budget=200)
        #         )
        #     )

    async def marking_criteria(self, new_criteria, student_composition, question_statement, student_class, student_word_count, criteria, language, rubric_examples = [], img_urls = [], is_ielts = False):
        score_detail = None
        breakdown, possible_scores = criteria_breakdown(new_criteria)
        # print(f"Breakdown:\n{breakdown}\nPossible Scores: {possible_scores}")
            
        response = await self.criteria_marking_new(question_statement, new_criteria, student_composition, student_word_count, student_class, breakdown, language, rubric_examples, img_urls, is_ielts)
        result_student = response["output"]

        feedback = {
            "criteria_name": new_criteria['name'],
            "feedback": result_student['feedback_simplified'],
            "feedback_detailed": result_student['feedback']
        }

        # print(f"Mark: {result_student['mark']}")

        ## updating final_score to only add the student mark without manipulating it regardless of the model mark
        score = max(min(possible_scores), min(max(possible_scores), result_student['mark']))
        score_detail = await get_score_detail(
            criteria = criteria, 
            score = score, 
            feedback = result_student['feedback_simplified'],
            feedback_detailed = result_student['feedback']
        )

        return {
            "feedback": feedback, 
            "score": score, 
            "score_detail": score_detail
        }

class EssayMarkingChain:
    @staticmethod
    async def check_rubric_interdependency(all_criteria, student_class, api_key):
        output_structure = r"""
            {
                "is_dependent": <boolean, True if any of the criteria specifically mentions another criteria, else False>
            }
        """

        prompt = """You are a {student_class} instructor. Your task is to assess if any of the criteria specifically mentions another criteria. \
        
        If there is only one criteria, return false.
        If the criteria is of IELTS writing test Task 1 or Task 2, return true.

        Your response should be in the following JSON format:
        {output_structure}
        """.format(
            student_class = student_class,
            output_structure = output_structure,
        )

        question = """Here are all of the criteria and their breakdowns:
        ```
        {criteria_breakdown}
        ```""".format(
            criteria_breakdown = "\n```\n".join(["Criteria name: {name}\nMax score: {max_score}\nBreakdown:\n".format(
                name = criteria['name'],
                max_score = criteria['max_score']) + criteria_breakdown_range_flatten(criteria) for criteria in all_criteria]
            )
        )

        model = "gpt-4o"
        response, usage, model = await get_completion_turbo(prompt, model, question, api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
        
        return json.loads(response), usage, model
    
    @staticmethod
    async def compo_marking_v3(question_statement, rubric_table, student_composition, model_composition, student_class, language, examples = [], total_tokens = 0, total_cost = 0, models = set(), api_key = None):
        """"
        question_statement: An HTML string that contains the question statement. All the <img> tags will \
        be replaced with <p> tags that contain the description of the image.
        rubric_table: A list of criteria that the student's composition will be assessed against. The \
        structure of the table is as follows
        {
            'rubric': [
                {
                    "name": "Content",
                    "max_score": 20,
                    "breakdown": [
                        {
                            "from_score": 0,
                            "to_score": 5,
                            "score_type": "range" or "fixed"
                            "description": "The student's composition does not contain any relevant information."
                        },
                        ...
                    ]
                },
                ...
            ]
        }
        student_composition: A string that contains the student's composition.
        model_composition: A string that contains the model composition. The model solution will have \
        the highest score for all the criteria.
        student_class: A string that contains the student's class. For example, "Primary 1".
        """
        final_score = 0
        feedback_list = []
        score_detail = []
        img_urls = []
        is_ielts = False

        preprocessor = EssayMarkingPreprocessor(
            total_tokens = total_tokens,
            total_cost = total_cost,
            api_key = api_key
        )

        ## preprocessing: prepare rubric_table and examples
        new_rubric_table = rubric_table
        all_rubric_examples = {(rubric["name"], rubric.get("rubric_id", "")): [] for rubric in rubric_table}
        for example in examples:
            for detail in example["score_detail"]:
                try:
                    rubric_key = (detail["name"], detail.get("rubric_id", "")) if (detail["name"], detail.get("rubric_id", "")) in all_rubric_examples.keys() else (detail["name"], "")
                except:
                    rubric_key = (detail["component"], detail.get("rubric_id", "")) if (detail["component"], detail.get("rubric_id", "")) in all_rubric_examples.keys() else (detail["component"], "")
                all_rubric_examples[rubric_key].append({"student_composition": example["student_composition"], "score": detail["score"]})

        ## preprocessing: clean up the question statement as well as checking if image link exists to extract the image description
        question_statement, img_urls = await preprocessor._preprocess_question_statement(question_statement)

        ## preprocessing: get word count from both essays
        student_word_count, model_word_count = await preprocessor._get_essay_word_count(
            student_composition = student_composition, 
            model_composition = model_composition
        )

        ## get final data from preprocessor
        preprocessor_data = preprocessor._get_llm_data()
        total_tokens += preprocessor_data["total_tokens"]
        total_cost += preprocessor_data["total_cost"]
        models.update(preprocessor_data["models"])

        ## set up evaluator instance
        evaluator = EssayMarkingEvaluator(
            total_tokens = total_tokens,
            total_cost = total_cost,
            api_key = api_key
        )

        ## If rubrics are from IELTS writing marking, indicate so
        if sorted([item['name'].lower() for item in new_rubric_table]) == ['coherence & cohesion', 'grammatical range & accuracy', 'lexical resource', 'task achievement']:
            is_ielts = True

        coroutines = [evaluator.marking_criteria(
            new_criteria = new_criteria,
            student_composition = student_composition,
            question_statement = question_statement,
            student_class = student_class,
            student_word_count = student_word_count,
            criteria = criteria,
            language = language,
            rubric_examples = rubric_examples,
            img_urls = img_urls,
            is_ielts = is_ielts
        ) for new_criteria, criteria, rubric_examples in zip(new_rubric_table, rubric_table, all_rubric_examples.values())]

        responses = await asyncio.gather(*coroutines)

        ## get final data from evaluator
        evaluator_data = evaluator._get_llm_data()
        total_tokens += evaluator_data["total_tokens"]
        total_cost += evaluator_data["total_cost"]
        models.update(evaluator_data["models"])

        for response in responses:
            final_score += response['score']
            feedback_list.append(response['feedback'])
            score_detail.append(response['score_detail'])

        feedback_string = ""
        feedback_string_detailed = ""
        for feedback in feedback_list:
            feedback_string += f"{feedback['criteria_name']}: {feedback['feedback']}\n"
            feedback_string_detailed += f"{feedback['criteria_name']}: {feedback['feedback_detailed']}\n"

        feedback_output = {
            "feedback": {
                "mark": final_score,
                "feedback": feedback_string,
                "feedback_detailed": feedback_string_detailed,
                "score_detail": score_detail
            },
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "models": list(models)
        }
        
        return feedback_output

    @staticmethod
    async def compo_marking_v3_combined(question_statement, rubric_table, student_composition, model_composition, student_class, language, examples = [], total_tokens = 0, total_cost = 0, models = set(), api_key = None):
        """"
        question_statement: An HTML string that contains the question statement. All the <img> tags will \
        be replaced with <p> tags that contain the description of the image.
        rubric_table: A list of criteria that the student's composition will be assessed against. The \
        structure of the table is as follows
        {
            'rubric': [
                {
                    "name": "Content",
                    "max_score": 20,
                    "breakdown": [
                        {
                            "from_score": 0,
                            "to_score": 5,
                            "score_type": "range" or "fixed"
                            "description": "The student's composition does not contain any relevant information."
                        },
                        ...
                    ]
                },
                ...
            ]
        }
        student_composition: A string that contains the student's composition.
        model_composition: A string that contains the model composition. The model solution will have \
        the highest score for all the criteria.
        student_class: A string that contains the student's class. For example, "Primary 1".
        """

        is_ielts = False
    
        preprocessor = EssayMarkingPreprocessor(
            total_tokens = total_tokens,
            total_cost = total_cost,
            api_key = api_key
        )

        ## preprocessing: clean up the question statement as well as checking if image link exists to extract the image description
        question_statement, img_urls = await preprocessor._preprocess_question_statement(question_statement)

        ## preprocessing: get word count from both essays
        student_word_count, model_word_count = await preprocessor._get_essay_word_count(
            student_composition = student_composition, 
            model_composition = model_composition
        )

        ## get final data from preprocessor
        preprocessor_data = preprocessor._get_llm_data()
        total_tokens += preprocessor_data["total_tokens"]
        total_cost += preprocessor_data["total_cost"]
        models.update(preprocessor_data["models"])

        ## prepare rubric_table
        new_rubric_table = rubric_table

        ## set up evaluator instance
        evaluator = EssayMarkingEvaluator(
            total_tokens = total_tokens,
            total_cost = total_cost,
            api_key = api_key
        )

        for idx, item in enumerate(rubric_table):
            item['id'] = idx
        for idx, item in enumerate(new_rubric_table):
            item['id'] = idx

        if sorted([item['name'].lower() for item in new_rubric_table]) == ['coherence & cohesion', 'grammatical range & accuracy', 'lexical resource', 'task achievement']:
            is_ielts = True
        elif sorted([item['name'].lower() for item in new_rubric_table]) == ['coherence & cohesion', 'grammatical range & accuracy', 'lexical resource', 'task response']:
            is_ielts = True
        else:
            is_ielts = False

        # print(f"is ielts: {is_ielts}")
            
        response_student = await evaluator.criteria_marking_new_combined(
            question_statement = question_statement, 
            rubric_components = new_rubric_table, 
            student_composition = student_composition, 
            word_count = student_word_count, 
            student_class = student_class, 
            language = language, 
            examples = examples,
            img_urls = img_urls,
            is_ielts = is_ielts
        )

        ## final step: calculate final score and construct feedback object
        feedback_object = await construct_essay_marking_feedback_object(
            essay_marking_result = response_student["output"],
            rubric_table = rubric_table,
            new_rubric_table = new_rubric_table
        )
        
        ## get final data from evaluator
        evaluator_data = evaluator._get_llm_data()
        total_tokens += evaluator_data["total_tokens"]
        total_cost += evaluator_data["total_cost"]
        models.update(evaluator_data["models"])
        
        ## construct feedback output
        feedback_output = {
            "feedback": {
                "mark": feedback_object["final_score"],
                "feedback": feedback_object["feedback_string"],
                "feedback_detailed": feedback_object["feedback_string_detailed"],
                "score_detail": feedback_object["score_detail"]
            },
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "models": list(models)
        }
        
        return feedback_output