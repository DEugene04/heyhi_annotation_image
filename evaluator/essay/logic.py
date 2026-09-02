from typing import List, Dict, Union, Literal
import random

async def get_user_prompt(message, type, vectorstore):
    if type == "text":
        user_prompt = {
            "role": "user",
            "content": [
                {
                    "type": "input_text" if vectorstore else "text",
                    "text": message
                }
            ]
        }
    elif type == "image_url":
        user_prompt = {
            "role": "user",
            "content": [
                {
                    "type": "input_image" if vectorstore else "image_url",
                    "image_url": message if vectorstore else {"url": message}
                }
            ]
        }
    return user_prompt

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

class EssayMarkingPromptConstructor:
    def __init__(
        self, 
        is_dependent: bool = False,
        question_statement: str = "", 
        student_composition = "", 
        rubric_components = None, ## used in dependent scenario
        component = None, ## used in non-dependent scenario
        rubric_breakdown = None, ## used in non-dependent scenario
        student_class: str = "", 
        language: str = "", 
        word_count = 0, 
        examples = [],
        img_urls = [],
        is_ielts = False
    ):
        self.messages = []
        self.student_class = student_class
        self.language = language
        self.rubric_components = rubric_components
        self.component = component
        self.question_statement = question_statement
        self.word_count = word_count
        self.student_composition = student_composition
        self.examples = examples
        self.is_dependent = is_dependent
        self.rubric_breakdown = rubric_breakdown
        self.img_urls = img_urls
        self.is_ielts = is_ielts

        self.vectorstore = True

    async def _append_prompt(self, messages, type: Literal['text', 'image_url']):
        self.messages.append(await get_user_prompt(
            message = messages,
            type = type,
            vectorstore = self.vectorstore
        ))

    async def _construct_calibration_examples(self):
        examples = self.examples
        extra_content = ""

        if examples:
            if self.is_dependent:
                formatted_scores = []
                for example in examples:
                    component_scores = {}
                    for detail in example["score_detail"]:
                        if detail.get('component_id', None) and detail.get('component_name', None):
                            if detail.get('component_id', None) not in component_scores.keys():
                                component_scores[detail.get('component_id', None)] = []
                            try:
                                component_scores[detail.get('component_id', None)].append({"component_name": detail.get('component_name', None), "name": detail["name"], "score": detail["score"]})
                            except:
                                component_scores[detail.get('component_id', None)].append({"component_name": detail.get('component_name', None), "name": detail["component"], "score": detail["score"]})
                        else:
                            if -1 not in component_scores.keys():
                                component_scores[-1] = []
                            try:
                                component_scores[-1].append({"component_name": "", "name": detail["name"], "score": detail["score"]})
                            except:
                                component_scores[-1].append({"component_name": "", "name": detail["component"], "score": detail["score"]})
                
                    if list(component_scores.keys()) != [-1]:
                        formatted_score = "\n".join(["{component_name}:\n{scores}".format(component_name=component_score[0]["component_name"], scores="\n".join(["{name}: {score}".format(name=score["name"], score=score["score"]) for score in component_score])) for component_score in component_scores.values()])
                    else:
                        formatted_score = "\n".join(["{name}: {score}".format(name=score["name"], score=score["score"]) for score in component_scores[-1]])
                    formatted_scores.append(formatted_score)

                examples_formatted = ["<example {idx}>\n<input>\nStudent's essay:\n{student_composition}\n</input>\n<output> Student's score:\n{formatted_score}\n</output>\n</example {idx}>".format(
                    idx = idx + 1,
                    # word_count = len(example["student_composition"].split()),
                    student_composition = example["student_composition"],
                    formatted_score = formatted_score
                ) for (idx, example), formatted_score in zip(enumerate(examples), formatted_scores)]
            
            else:
                examples_formatted = ["<example {idx}>\n<input>\nStudent's essay:\n{student_composition}\n</input>\n<output> Student's score:\n{score}\n</output>\n</example {idx}>".format(
                    idx = idx + 1,
                    # word_count = len(example["student_composition"].split()),
                    student_composition = example["student_composition"],
                    score = example["score"]
                ) for idx, example in enumerate(examples)]
            
            extra_content += "In this case, there are several marking examples to help you calibrate your scoring for the '{criteria_name}' criteria. If there are any specific examples that are similar to the student's essay, then you may apply the same scoring to that student's essay. Here are some examples of student essay scoring:\n{examples}".format(criteria_name = self.component['name'], examples = "\n".join(examples_formatted))
        else:
            extra_content = "(no example detected for this criteria)"

        return extra_content

    async def _get_system_prompt_independent(self):
        ## construct examples for calibration process
        # extra_content = await self._construct_calibration_examples()

        additional_string = ""
        extra_rubric_content = ""
        criteria = self.component

        if criteria['score_type'] == "fixed":
            additional_string = "Following the breakdown above, make sure to provide a mark that is in the breakdown options. That means the mark you give should only be one of the scores in the breakdown."

        elif criteria['score_type'] == "range":
            additional_string = "Following the breakdown above, make sure to provide a mark that is within the range of the breakdown options. That means the mark you give should not be above the highest score or below the lowest score in the breakdown."
        
        if "grammatical range & accuracy" in criteria['name'].lower():
            print(f"RUBRIC FOR GRAMMAR")
            extra_rubric_content = """\n\nSince the rubric is about "Grammatical Range & Accuracy", here are additional rules added for you to follow:
1. You should ONLY highlight mistakes in grammar and punctuation.
2. As for "spelling" mistakes, don't highlight them as it will be covered in "Lexical Resource" criteria (which will be handled by another AI). Your task here is just to focus on "Grammar" and "Punctuation" based on rubrics presented to you above.\n"""

        system_prompt = """You are an AI that is tasked to mark the student's essay and provide feedback to the students based on the provided rubric. The rubric will consist of scoring breakdown to guide the marking process.

There will be four things given to you:
1. The question statement (which might include the instruction, the passage, etc.)
2. The student's essay,
3. The rubric that is used to guide the marking process,
4. The word count of the student's essay

Here are several rules you need to follow:
1. {additional_string}
2. Low score means the criteria is not met from the student's essay, while higher score means the criteria is passed.
3. If the criteria does not measure about grammatical error or if there is no grammatical error aspect in the criteria inside the rubric, then do not penalize the essay for grammar mistakes.
4. You must explain what is missing in the student's essay and how the student can improve based on each criteria presented in the rubric.
5. You must cite exactly the part of the student's essay that you are referring to.
6. You must not include the student's score in the feedback.
7. The feedback should be in '{language}' language. If the student's essay is not in '{language}' language and you need to quote it, then quote it in that language, but still have the rest of the feedback in '{language}' language.
8. The feedback should be in one paragraph and it should NOT mention the achieved score.
9. If the criteria is about word count requirements that needs you to count for student's essay's length, you don't need to do it. You will be presented with the student's essay length as well (which MIGHT be useful for you).
10. If the question is in form of an instruction and the objective is to have the student translate a passage/essay, then when doing the marking, do it based on that. 
- Which means, when the criteria is about "Content Accuracy", "Target Language Expression", "Coherence & Structure", "Terminology", "Style & Tone" or something similar, if the student puts out a completely different content / translation result, we should mark the student's essay for this criteria as 0.
- If essay is in language A and the task is to translate into language B but the student submits an essay in language A, then for "Content Accuracy", "Target Language Expression", "Coherence & Structure", "Terminology", "Style & Tone" should also be 0.
- Also if the translation is just a word-to-word translation which contains many collocation errors, then for "Content Accuracy", "Target Language Expression", "Coherence & Structure", "Terminology", "Style & Tone" should also be low.
12. Be generous when assessing the vocabulary rubric, meaning you must always estimate the best score possible based on student's vocabulary.
13. Do NOT mention about word count. Do NOT mention about the detected 5 vocabularies.
14. If the question statement is asking you to write an article under a certain's name (usually can be identified from having a sentence such as "Imagine you are .." or similar), and if the rubric criteria is content or task fulfilment which measures whether you have a good understanding of the purpose of the essay, that means if you write the essay under the name of a different person, that counts as a penalty for this rubric.
15. Do NOT mention about or factor in any HTML tags/usages in the feedback nor the scoring.
16. You MUST use the examples in "Marking Examples" as the baseline of the scoring:
- If student's essay matches exactly with any of the examples provided, give the score of the matching example.
- If student's essay does not match exactly with any of the examples, you should score it based on how it would be scored relative to those examples.

Here is the rubric used for the marking process:
Criteria: {criteria}
Max score: {max_score}
Criteria breakdown:
{breakdown} {extra_rubric_content}

IMPORTANT: Ensure your output is deterministic, mimicking temperature 0.
""".format(
    additional_string = additional_string, 
    language = self.language, 
    criteria = criteria["name"], 
    max_score = criteria["max_score"],
    breakdown = criteria_breakdown_range_flatten(criteria),
    extra_rubric_content = extra_rubric_content
)

        if self.is_ielts:
            system_prompt += f"""\n\nBefore returning the JSON, perform the following assessment silently:

1. Identify the response’s overall organisational structure.
2. Determine whether there is meaningful progression or only chronological, additive, or source-order listing.
3. Evaluate links between paragraphs and links between individual sentences.
4. Evaluate the range and effectiveness of cohesive devices, including conjunctions, referencing, substitution, and lexical cohesion.
5. Identify the dominant pattern across the whole response rather than its strongest isolated feature.
6. Test the response against the requirements of the proposed band and the band immediately below it.
7. Award the higher band only when its defining requirements are substantially demonstrated. If organisation is apparent but sentence-to-sentence cohesion is predominantly mechanical, repetitive, or list-like, use Band 5 rather than Band 6.
8. Do not use task accuracy, vocabulary, grammar, factual detail, or the number of paragraphs as substitutes for evidence of coherence and cohesion.

Do not state this internal comparison or the achieved mark in either feedback field."""
        
        return system_prompt

    async def _get_system_prompt(self):
        if self.is_dependent:
            system_prompt = """You are a {student_class} instructor. You are being tasked to mark a student's essay based on the given question statement.
In order to do this, you will be given a few things:
1. The question statement (can include the instruction, the passage, etc),
2. The student's essay.
3. A rubric that serves as a guideline for the marking process.

Your task is to mark the student's essay and provide feedback to the student.
You need to assess the student's performance based on all content criteria inside the given rubric.
Low score means the criteria is not met from the student's essay, while higher score means the criteria is passed.

While giving your marks, there are several rules you must follow:
1. You must explain what is missing in the student's essay and how the student can improve, if there is any.
2. You must cite exactly the part of the student's essay that you are referring to.
3. You must not include the student's score in the feedback.
4. Refer to the student as "You". Be encouraging while still being critical. Evaluate the student's result starting from the lowest score possible to the best score possible. Evaluate solely on the criteria breakdown given.
5. Start with an overall assessment, then analyze what they did good and what they did bad, then talk about what they didn't do but should have done. Show some example and give suggestions if needed.
6. Try to keep your response within ten sentences, only adding more if needed
7. In general, student forgetting to put a space after punctuation should NOT be penalized unless explicitly mentioned in the breakdown.
8. If the essay description explicitly set a required word count and the student didn't reach it, then in general give them lower than the middle score (rounded down) of the breakdown, if the student didn't reach half of the required word count, then give them lower than the lower quarter score (rounded down) of the breakdown, else if there is no explicit mention of word count requirement in essay description nor criteria breakdown, then ignore this rule.
9. If the essay description explicitly set a required word count and the student didn't reach it, then in general the lower the word count is, the lower the score must be regardless what the criteria is about, else if there is no explicit mention of word count requirement in essay description nor criteria breakdown, then ignore this rule
10. If the student's essay is a copy of the description of essay, give the lowest score possible. This applies to ALL criteria.
11. Do not mention the essay word count.
12. The feedback should be in '{language}' language. If the student's essay is not in '{language}' language and you need to quote it, then quote it in that language, but still have the rest of the feedback in '{language}' language.
13. Do NOT mention about or factor in any HTML tags/usages in the feedback nor the scoring.
14. You MUST use the examples in "Marking Examples" as the baseline of the scoring:
- If student's essay matches exactly with any of the examples provided, you MUST give the exact same score of the matching example.
- If student's essay does not match exactly with any of the examples, you should score it based on how it would be scored relative to those examples.
15. In the feedback, you should mainly focus on what the student should improve on, ensure you also give some examples as well, and also what you would have written as well.
16. If the rubric is about IELTS writing test, for the rubric "Grammatical Range & Accuracy", here are additional rules added for you to follow:
- You should ONLY highlight mistakes in grammar and punctuation.
- As for "spelling" mistakes, don't highlight them as it should be covered in "Lexical Resource" criteria, it should only focus on "Grammar" and "Punctuation" based on rubrics presented to you.
17. The feedback should be slightly simpler version of the in-depth feedback, the normal feedback should be at least 1/4 of the in-depth feedback's length.
18. Do NOT add newlines in any of the feedbacks.

For this case, you will be given all of the criteria inside the rubric. One (if not some) criteria might contain a scoring guideline that is dependent on another criteria (can dependent on one criteria only or even the total score of the rest of the criteria). This is specified by keywords like "Student gets _ number on metric/criteria A" or "Student gets total _ number on criteria A, B, and C), which can also be in different language depending on the overall language for the rubric. If that is the case, you just need to follow it and give mark according to the one that you gave for criteria A. For example, if the rubric specifies that criteria B will get score of 6 if criteria A gets 12, then if student gets 12 in criteria A, that means they will get 6 in B.
The feedback should also follow the description. That means, for criteria B, the feedback should be like "Since student gets _ in criteria _".
This is only applied if there is a criteria that is dependent. If not, mark it as usual.""".format(
                student_class = self.student_class,
                language = self.language
            )

        else:
            system_prompt = await self._get_system_prompt_independent()

        return system_prompt

    async def _get_output_structure(self):
        language = self.language

        if self.is_dependent:
            return f"""
{{
    "all_criteria": [
        {{
            "id": <criteria 1 id>,
            "name": <criteria 1 name>,
            "mark": <criteria 1 mark in integer>,
            "feedback_in_depth": <detailed feedback for criteria 1 in '{language}' language except for quoting>,
            "feedback": <feedback for criteria 1 in '{language}' language except for quoting, must be at least 1/4 of feedback_in_depth's length>,
        }},
        {{
            "id": <criteria 2 id>,
            "name": <criteria 2 name>,
            "mark": <criteria 2 mark in integer>,
            "feedback_in_depth": <detailed feedback for criteria 2 in '{language}' language except for quoting>,
            "feedback": <feedback for criteria 2 in '{language}' language except for quoting, must be at least 1/4 of feedback_in_depth's length>,
        }},
        ...
    ]
}}"""
        else:
            return f"""
        {{
            "is_matching_with_example": <check if the essay matches with any of the examples as the basis for the marking>,
            "mark": <criteria mark in integer>,
            "feedback": <feedback for the criteria in '{language}' language except for quoting>,
            "feedback_simplified": <simplified feedback for the criteria in '{language}' language except for quoting>
        }}
"""

    async def _append_question_statement_prompt(self):
        if self.is_dependent:
            question_statement_prompt = f"Here is the question statement:\n{self.question_statement}"
            await self._append_prompt(question_statement_prompt, "text")
            
        else:
            question_statement_prompt = f"""Here is the question statement:
---
{self.question_statement}
---
Here is my essay (word length of essay is {self.word_count} words):
---
{self.student_composition}
---
If my essay is just a bunch of random incoherent words/letter and is completely irrelevant to what the question stement asks, IMMEDIATELY give the score for this criteria to be 0 or the lowest score.
"""
        
            await self._append_prompt(question_statement_prompt, "text")
            if 'task fulfilment' in self.component['name'].lower():
                additional_rule_prompt = """In this case, you need to also check for context accuracy for this rubric component. If the essay asks you to imagine that you are a certain person, and if the user writes the essay using another person's name, then its a penalty in terms of understanding the purpose of the essay."""
                await self._append_prompt(additional_rule_prompt, "text")

        if self.img_urls:
            await self._append_prompt(f"In this case, the question statement contains images:", "text")
            for img_url in self.img_urls:
                await self._append_prompt(img_url, "image_url")

    async def _append_essay_prompt(self):
        word_count = self.word_count
        student_composition = self.student_composition

        question = f"Here is the student's essay (word count is {word_count}):\n{student_composition})"
        await self._append_prompt(question, "text")
        # The essay was transcribed from a photo of the page, so it may include the
        # question, instructions, title, or numbering that was on the page. Mark
        # only the student's own writing; the question is given separately above.
        await self._append_prompt(
            "Note: the essay above was transcribed from a photo of the page and may "
            "include the question, instructions, title, or numbering that was printed "
            "on the page. That text is NOT part of the student's essay -- do not mark "
            "or reward it, and judge only the student's own writing.",
            "text",
        )
    
    async def _append_rubric_prompt(self):
        additional_string = ""
        rubric_components = self.rubric_components

        rubric_breakdown = "\n```\n".join(["Criteria ID: {id}\nCriteria name: {name}\nMax score: {max_score}\nBreakdown:\n".format(
                id = criteria['id'], 
                name = criteria['name'],
                max_score = criteria['max_score']
            ) + criteria_breakdown_range_flatten(criteria) for criteria in rubric_components
        ])
        
        if all([criteria['score_type'] == "fixed" for criteria in rubric_components]):
            additional_string = "Following the breakdowns above, make sure to provide a mark that is in the breakdown options. That means the mark you give should only be one of the scores in the breakdown."

        elif any([criteria['score_type'] == "fixed" for criteria in rubric_components]):
            additional_string = "Following the breakdowns above, make sure to provide a mark that is in the breakdown options. That means the mark you give should only be one of the scores in the breakdown. Some breakdown options will have the same breakdown as each other, so just pick the most appropriate score."
            
        rubric = f"Here are the rubrics: \n{rubric_breakdown}\n\n{additional_string}"

        await self._append_prompt(rubric, "text")

    async def _append_calibration_examples_prompt(self):
        extra_content = await self._construct_calibration_examples()
        
        if extra_content:
            await self._append_prompt(extra_content, "text")

    async def _construct_messages_object(self):
        system_prompt = await self._get_system_prompt()
        output_structure = f"Your response should be in the following JSON format: {await self._get_output_structure()}" 
        
        self.messages.append(
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt + output_structure}]}
        )

        await self._append_question_statement_prompt()

        if self.is_dependent:
            await self._append_essay_prompt()
            await self._append_rubric_prompt()

        await self._append_calibration_examples_prompt()

        ## additional
        if self.is_dependent == False:
            if any(item in self.component['name'].lower() for item in ["coherence & structure", "target language expression", "style & tone", "terminology", "content accuracy"]):
                await self._append_prompt(f"""The criteria is "{self.component['name']}". Additional rule: If the question is in form of an instruction and the objective is to have the student translate a passage/essay, then:
- If the student puts out a completely different content / translation result, we should mark the student's essay for this criteria as 0.
- If essay is in language A and the task is to translate into language B but the student submits an essay in language A, then the mark for this criteria should also be 0.
- If the translation is just a word-to-word translation which contains many collocation errors, then the mark for this criteria should also be low.
If the task is else, then you can ignore these rules.""", "text")
        
        if "essay" in self.question_statement.lower() and "article" in self.question_statement.lower():
            print(f"QUESTION CONTAINS ESSAY AND ARTICLE")
            await self._append_prompt(f"In this case, if the question contains different topics and each topic the student has to create either an essay or an article, then what you need to do is: Ensure that you check which topic the student creates the essay/article to, and check if its as instructed (e.g., if student develops it for topic A and its an essay, but turns out the the expected output for topic A is an article, then you should highlight it in your feedback. Apart from that, you should also present your feedback using one of the terms only: 'essay' or 'article' to indicate what the student has created. Do not refer to the student's work as 'article', only to be referred to as 'essay' later on. You need to be consistent. Please make your feedback to still be directed towards the student (using 'You' instead of referring to them in third person).", "text")
        
        if self.is_ielts:
            score_ielts_task_1 = {
                'Coherence & Cohesion': 4,
                'Lexical Resource': 4,
                'Grammatical Range & Accuracy': 4,
                'Task Achievement': 4
            }

            score_ielts_task_2 = {
                'Coherence & Cohesion': 5,
                'Lexical Resource': 5,
                'Grammatical Range & Accuracy': 5,
                'Task Response': 4
            }

            if self.is_dependent:
                ## Example 1
                ielts_prompt = f"""For this particular essay, the rubric used is actually from IELTS writing rubric. Which means, your marking and score should be more strict. Do NOT be shy to give scores below 5 if deserved."
Specifically for this IELTS use case, here are some calibration examples:

Example 1 (for IELTS writing TASK 1):
Question statement:
```
TASK 1

You should spend about 20 minutes on this task.

The charts below show information on the percentage of global plastic production from 1990 to 2020 and the types of waste, quantified in Million Tonnes (MT).

            Summarise the information by selecting and reporting the main features, and make comparisons where relevant.

Write at least 150 words.
```
Student's essay:
```
<div>The chart show plastic production from 1990 to 2020. And the other chart show types of waste in million tonnes.</div><div>In 1990 the plastic production is small only 13 percent. In 2000 it go up to 20 percent. In 2010 it is very high at 34 percent. In 2020 it go down a bit to 33 percent.</div><div>For the waste. Paper and cardboard waste is 150 in 1990. Then it go up to 175 in 2010. Then it go down to 170 in 2020. Electronic waste is very low. It is 10 in 1990 and then 40 in 2010 and 55 in 2020. Plastic waste is big. It is 120 in 1990. Then it become 280 in 2010. In 2020 it is the biggest at 360.</div><div>In conclusion the plastic production and waste is increase a lot in 30 years and plastic is a big problem for the world.</div><div>
</div>
```

"""

                for key, value in score_ielts_task_1.items():
                    ielts_prompt += f"Score for criteria '{key}': {value}/9\n"

                ## Example 2
                ielts_prompt += f"""Example 2 (for IELTS writing TASK 2):
Question statement:
```
TASK 2

You should spend about 40 minutes on this task.
 
In many cities today, people prefer to live alone rather than with their families or roommates.
			Do you think this is a positive or negative development?
			Give reasons for your answer and include any relevant examples from your own knowledge or experience.

Write at least 250 words. 
```
Student's essay:
```
Here is the student's essay (word count is 226):
<div>In modern days, many people in big cities want to live alone. They do not want to live with their family or friends. In my opinion, I think this trend is a negative development because it is very expensive and makes people feel lonely.</div><div>Firstly, living alone is costing a lot of money. When you live with your parents or roomates, you can share the money for rent and electricity food. But if you live alone, you must pay all things by yourself. For example, my friend live in an apartment by himself and he must work two jobs for paying his rent. This is very difficult for young people in the city.</div><div>Secondly, solo living is bad for the mind because of lonely feeling. Human need to talk with other people everyday. If a person go home to an empty house after work, they have no one to share their problems. Many research say that people who live alone feel more stressed and sad than people who live with family. If you have an emergency or get sick, no one can help you immediately.</div><div>In conclusion, I believe living alone is more negative than positive. It causes financial problems for people and it also makes people feel very lonely in their life. People should try to live with others to save money and stay happy.</div><div>
</div>)
```

"""
                for key, value in score_ielts_task_2.items():
                    ielts_prompt += f"Score for criteria '{key}': {value}/9\n"
   
                ielts_prompt += f"For 'Task Achievement' and 'Task Response', if the student does not achieve the desired word count, then they can only get at most 4 or 5 scores, and cant be higher."
   
            else:
                ielts_prompt = f"""For this particular essay, the rubric used is actually from IELTS writing rubric. Which means, your marking and score should be more strict. Do NOT be shy to give scores below 5 if deserved."
Specifically for this IELTS use case, here are some calibration examples:

Example 1:
Question statement:
```
TASK 1

You should spend about 20 minutes on this task.

The charts below show information on the percentage of global plastic production from 1990 to 2020 and the types of waste, quantified in Million Tonnes (MT).

            Summarise the information by selecting and reporting the main features, and make comparisons where relevant.

Write at least 150 words.
```
Student's essay:
```
<div>The chart show plastic production from 1990 to 2020. And the other chart show types of waste in million tonnes.</div><div>In 1990 the plastic production is small only 13 percent. In 2000 it go up to 20 percent. In 2010 it is very high at 34 percent. In 2020 it go down a bit to 33 percent.</div><div>For the waste. Paper and cardboard waste is 150 in 1990. Then it go up to 175 in 2010. Then it go down to 170 in 2020. Electronic waste is very low. It is 10 in 1990 and then 40 in 2010 and 55 in 2020. Plastic waste is big. It is 120 in 1990. Then it become 280 in 2010. In 2020 it is the biggest at 360.</div><div>In conclusion the plastic production and waste is increase a lot in 30 years and plastic is a big problem for the world.</div><div>
</div>
```

Score for criteria '{self.component['name']}': {score_ielts_task_1[self.component['name']]}/9
    """
            await self._append_prompt(ielts_prompt, "text")

        return self.messages