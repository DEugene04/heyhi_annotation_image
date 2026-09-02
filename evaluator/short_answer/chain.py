from pydantic import BaseModel, field_validator

from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request, HTTPException
from tools import utils

from typing import List, Dict, Union

import os, json
from json import JSONDecodeError
import re
from bs4 import BeautifulSoup

from openai import OpenAI, AsyncOpenAI
import time
import asyncio

from decimal import Decimal
from typing import Any

from evaluator.short_answer.logic import AutoMarking, QuestionContext, AnswerContext, RubricContext, QuestionMarkContext, AutomarkingUserPromptConstructor, AutomarkingSystemPromptConstructor, MarkingResultContext
from evaluator.short_answer.logic import QuestionExtractor, RubricTransformer
from evaluator.dynamic_llm_call import DynamicAsyncOpenAI

import config

class AutoMarkingChain:
    @staticmethod
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

    @staticmethod
    async def get_system_prompt_automarking(
        question_context: QuestionContext
    ) -> str:
        """
        Get Automarking System Prompt
        """

        system_prompt_constructor = AutomarkingSystemPromptConstructor(
            question_context = question_context
        )

        system_prompt = await system_prompt_constructor._construct_system_prompt()

        return system_prompt

    @staticmethod
    async def get_user_prompt_automarking_single_answer(messages, question_context: QuestionContext):
        """
        Get the user prompt for automarking with single answer (Open Ended Questions or Fill in the Blank using marking scheme = "Per Blank")
        """
        user_prompt_constructor = AutomarkingUserPromptConstructor(
            messages = messages, 
            question_context = question_context,
            is_multi = False
        )

        messages = await user_prompt_constructor._construct_messages_object()
            
        return messages

    @staticmethod
    async def get_user_prompt_automarking_multi_answer(messages, question_context: QuestionContext, marking_results):
        """
        Get the user prompt for automarking with multiple answers (Fill in the Blank with multiple blanks and using marking scheme = "Per Question")
        """
        user_prompt_constructor = AutomarkingUserPromptConstructor(
            messages = messages, 
            question_context = question_context,
            is_multi = True
        )

        messages = await user_prompt_constructor._construct_messages_object(
            marking_results = marking_results
        )

        return messages

    @staticmethod
    async def get_openai_message_multi_answer(question_context: QuestionContext, marking_results):
        """
        Combine AI Response from Multiple Answers into One 
        """
        vectorstore = False
        ## Get system prompt
        system_prompt = f"""You are an assistant that is responsible on acting as an automarker system. You will be marking a Fill in the Blank questions that can have multiple blanks inside one question.

To give you a background:
1. There is another AI system based on LLM called AI automarking system which can mark students answers on a question.
2. For Fill in the Blank questions with multiple blanks to be filled, there are two marking scheme available: "Mark per Blank" and "Mark per Question". 
-- If it is "Mark per Blank", the marking will be done per blank's answer. So if there are 2 blanks in one question, when the AI marks for blank 1's answer, it will not have any context about blank 2's answer.
-- If it is "Mark per Question", the marking will be done by marking all the blank's answer at once. This means, the AI will have all the contexts for all blanks answers.
3. In this case, you will be doing the second scheme, which is "Mark per Question", but with a little bit of twist. To ensure accuracy, the case will be treated as if it were using "Mark per Blank", which will get all the marking results for all the blanks, in which you will combine them into one.

To do this, you will be given:
1. The question that is being asked, included with indicator such as (blank 1) to indicate the blanks to be filled in the question
2. The student's answers as well as the previous AI marking result (along with the reasoning)
3. Rubric (optional): The rubric that is being used to do the marking (if there are any)
4. Incorrect Flags (optional): answers/terms to mark as strictly incorrect.
5. Marking Note (optional): question-specific marking exceptions or instructions.
6. Maximum score (provided by user)
7. Correct answer (Ground truth) and PAS answer (optional). PAS answer is an answer that has been annotated with a mark given by another AI. We call this as "Point Allocated System (PAS) answer". User will tell you whether "PAS answer" exists or not. Ground Truth Answer, however, is absolute and will always exist.

Default score increment is 0.5, unless the rubric given by the user states that no 0.5 marks are allowed. Lowest score is 0.

Based on those informations, here's what you need to do:
1. Read the AI marking results for each answer as well as their reasoning.
2. IMPORTANT: Look at the rubric given by the user (if there are any). There are some cases where the rubric talks about penalizing certain criteria (e.g. Grammar, Punctuation, Transference Errors, etc.). The rubric will be categorized into Full Mark, Half Mark, and Zero Mark. For example, let's say the criteria for the answer to be marked as zero marked is "having 3 or more transference errors". Since the previous AI marked using the "Mark per Blank" schema, that means this rule is applied to each answer (meaning if blank 1's answer has 2 transference errors and blank 2's answer has 1 transference error, then both would not be marked as zero mark, since it each of them are still below 3 transference error). But now, since you will be marking using "Mark per Question" scheme, you HAVE to combine the amount of transference errors then apply the rubric to the whole question. Using the same example, since blank 1 has 2 transference errors and blank 2 has 1 transference error, when combined, they will have 3 transference errrors, which in return would cause ALL answers to be zero mark.
3. Which means, if there is a rubric that is applied to the whole question, and ONE of them got 0 mark because of it, most likely the combined mark will be 0 as well, even though the other blank got full marks.
4. To better understand the mistakes categorization for English Sentence Transformation tasks, here are some rules:

## Mistakes Categorization
**English**
- General Rules (if no Rubric provided by the user, can follow these rules):
1. Any mistakes in punctuation should cause the answer to be 0. For example: ending the answer in a comma instead of a period. However, not ending the answer with a period is fine and should not be penalized, as long as it is not ended with a comma.
2. Different choice of wordings will not cause in score deduction. For example, the question asks to transform the sentence "The baby is cute. Everyone adores her". The PAS answer is "adores the baby's cuteness" and the ground truth answer is "adores the baby because she is cute". If the student answer is "adores the baby as she is cute" or "adores the baby since she is cute", it will still be considered as correct since "because" is similar to "as" and "since" in this context and does not change the meaning.
3. However, if it's not different choice of wordings but MISSING a word instead, this should be highlighted and be explained in your reasoning. An example of missing word is: The correct answer is "Unless it stops raining, we are not going to John’s house today" but the student answers "Unless it stops raining, we are not going to John's house", missing the word "today", which is crucial for Sentence Transformation questions and affects the meaning.

- Special rules (use this if there is a Rubric provided by the user and specifically talks about the category of mistakes). Can override the general rules.
1. Another aspect of marking Synthesis and Transformation or Sentence Transformation questions is transference errors. Transference errors are errors that does not affect grammar or meaning. Transference errors typically consist of SPELLING OR PUNCTUATION ERROR ONLY.
2. Examples of transference errors: 1) contractions, 2) spelling error that does not form a different word, and 3) punctuation error (misplacing comma, period, etc). These errors are not considered as changing the grammar/meaning, but if the rubric penalizes for transference error, then you should follow the marking based on how the rubric would penalize for the occurence of transference errors.
3. If the rubric mentions about penalizing transference errors that reach up a certain amount (for example 3 transference errors), count them by combining the amount of transference errors made in all the student answers for all blanks.
4. If the student adds a comma in a place that is not in the ground truth answer, it doesn't change the meaning or grammar, but it is considered as 1 transference error. 
5. Contractions, if not similar to the provided correct answer/ground truth, are counted as 1 transference error since it does not change the grammar/meaning and should follow the rubric given rubric about the marking guide for the mistake. Example of contractions: using "isn't" when the ground truth uses "is not", using "I'd" when the ground truth uses "I would", "they're", "we'll", etc.
6. However, there are also another kind of mistakes that changes the grammar/meaning. To check for grammar mistakes for this case, you should properly check with the correct answer/ground truth to see the differences. Differences to be checked are: missing words, transformed words, etc. If there are mistakes in these, then you should follow the given rubric (if there are any) to indicate what to do. Here are several examples of it (along with the case study example):
-- Punctuation errors like changing/removing apostrophe is counted as grammar error/changing the grammar. If that's the case, follow the given rubric (if there are any) about the marking guide for grammar changes in the transformed sentence. An example of punctuation errors that change the grammar is: Instead of "Unless it stops raining, we are not going to John’s house today", the student answers "Unless it stops raining, we are not going to Johns house today", changing "John’s" to Johns"
-- Mistakes in time expressions are considered also as grammatical mistake, which completely changes the meaning, and is considered as missing words that should be transformed from the original sentence. For example, if the transformed sentence required you to use "that night" instead of "tonight", using "tonight" will be counted as that error. Same goes to other time expressions. Another example is using "this" instead of "that" in certain cases. The marking should follow the rule stated by the rubric (if there are any). For example, if the rubric for zero mark states that "The transformed sentence is missing words that should be transformed from the original sentence", that means using "tonight" instead of "that night" is not counted as simple "transference error" anymore since it changes the meaning and the word "that night" is missing from the expected answer.
-- The difference in "this" and "that" in for synthesis and transformation or sentence transformation are considered as change of meaning and is not acceptable. For example, if the original sentence is "'I want this particular book for my birthday,' Nora told her parents" and the transformed sentence is "Nora told her parents that she wanted this particular book for her birthday", the usage of "this" here should be counted as a mistake since in the reported speech, using "that" will retain the meaning instead of using "this". In this case, compare the exact word used by the correct answer/ground truth. If the usage is different, then you can deem it as incorrect. Another example is that if the correct answer is "Either the old car or the new motorbikes are parked in the driveway" and the student answers "Either the old car or the new motorbikes are parked in that driveway", we can call it as incorrect since using "that" instead of "the" is incorrect and should be marked according to the rubric for this mistake.
-- Missing a phrase/word that changes the meaning slightly will still be considered as incorrect (if the rubric suggests so), even though the rest of the sentences are gramamtically correct and structurally sound. For example, if the student answers "Ling wondered why she had not returned the book" instead of "Ling wondered why she had not returned the book on time", the omission of phrase "on time" affects the meaning and should be treated according to the rubric, regardless of whether the rest of the sentences are correct or not.
-- Another examples of grammar errors: changing 'raining' to 'rain', 'houses' to 'house', changing 'John’s' to 'Johns', using 'race car' instead of 'racing car' (which changes the noun phrase)
-- Another examples of errors that forms a different word: changing 'deep' to 'dip', 'very' to 'vary'
-- For the usage of comparative structures, you have to follow the one that is used in the ground truth. For example, if the ground truth uses the "prefer X to Y" structure, that means the student also needs to use "prefer X to Y" structure. If the student uses "prefer X over Y" or "prefer X more than Y" instead, it should be considered as incorrect and be considered as incorrect grammar. This mistake is not considered as transference error and is considered as grammatically incorrect and changes the meaning of the sentence.
-- Omission of comparative structure words is also considered as grammatical error which changes the meaning. For example, if the transformed sentence is "Sharon prefers watching plays to movies" instead of "Sharon prefers watching plays to going to the movies", the omission of "going to the" here is not something that is minor since it changes the meaning. Which means, it is considered as missing a word that should be transformed and should be marked as 0.

When giving your response, in this case, give reasoning for all the answers (no need to be in numbered list, can be in the same paragraph). When doing this, you should align with your given mark and do not contradict.

Return your response in the following JSON format:
{{
    "student_answer": ["answer 1", "answer 2", etc.],
    "mark": final score of the answers, following the minimum and maximum mark as well as the increment.
    "reason": the reason behind the mark allocation for the student's answer, based on the given correct answer.
}}
"""

        messages = []
        system_prompt_object = {"role": "system", "content": [{"type": "input_text" if vectorstore else "text", "text": system_prompt}]}
        messages.append(system_prompt_object)
        
        messages = await AutoMarkingChain.get_user_prompt_automarking_multi_answer(
            messages = messages,
            question_context = question_context,
            marking_results = marking_results
        )

        return messages

    @staticmethod
    async def get_openai_message_single_answer(question_context: QuestionContext):
        """
        Get the OpenAI message for the AI Automarking Model. 
        Currently it is focused on Fill in the Blank (Synthesis and Transformations, Filling Tables) questions and Open Ended Questions.
        """
        vectorstore = False

        ## Get system prompt
        system_prompt = await AutoMarkingChain.get_system_prompt_automarking(
            question_context = question_context
        )

        messages = []
        system_prompt_object = {"role": "system", "content": [{"type": "input_text" if vectorstore else "text", "text": system_prompt}]}
        messages.append(system_prompt_object)
        
        messages = await AutoMarkingChain.get_user_prompt_automarking_single_answer(
            messages = messages,
            question_context = question_context
        )

        return messages

    @staticmethod
    async def get_openai_responses(
        messages, 
        type_multi: bool = False, 
        subject: str = "",
        question_context: QuestionContext = None,
        model: str = "",
        debug_mode: bool = False,
        auto_fail: bool = False,
        api_key: str = None
    ):
        try:
            if model == "":
                if subject == "IGCSE Business" or type_multi:
                    model = "gpt-4.1"
                else:
                    model = "gpt-4.1-mini"

                try:
                    if "typo" in question_context.marking_note.lower():
                        model = "gpt-4.1"
                except:
                    pass

            print(f"Model used: {model}")

            client = DynamicAsyncOpenAI(api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
            response = await client.chat.completions.create(
                model = model,
                messages = messages,
                # temperature = 0,
                # max_tokens = 2048,
                # top_p = 1,
                extra_body = {"debug_mode": debug_mode, "auto_fail": auto_fail},
                response_format = {
                    "type": "json_object"
                }
            )
        except Exception as e:
            if auto_fail:
                raise e
            print(f"Error in getting OpenAI Response found:\n{e}")
            if "quota" in str(e).lower():
                print(f"Quota issue. Please check. Using DeepSeek now...")
                client = DynamicAsyncOpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
                response = await client.chat.completions.create(
                    model = "deepseek-v4-flash",
                    messages = messages,
                    # temperature = 0,
                    # max_tokens = 2048,
                    # top_p = 1,
                    extra_body = {"thinking": {"type": "disabled"}},
                    response_format = {
                        "type": "json_object"
                    }
                )

        return {
            "response": response,
            "model": response.model
        }

    @staticmethod
    async def get_result_multi_answer(
        question_context: QuestionContext, 
        reqs: AutoMarking,
        debug_mode: bool,
        auto_fail: bool,
        api_key: str
    ):
        coroutines = []
        mark = question_context.mark_context
        answer_context = question_context.answer_context
        
        for index, (item_index, item_student_answer) in enumerate(zip(mark.fitb_index, answer_context.student_answer)):
            correct_answer_new, num_blanks, unique_blanks = await QuestionExtractor(reqs = reqs).extract_blank_section_dispatch(answer_context.original_correct_answer, item_index)

            mark_context_new = QuestionMarkContext(
                full_mark = mark.full_mark,
                step = "0.5",
                fitb_index = item_index,
                num_blanks = num_blanks,
                unique_blanks = unique_blanks,
                type_fitb = mark.type_fitb
            )

            answer_context_new = AnswerContext(
                correct_answer = correct_answer_new,
                original_correct_answer = answer_context.correct_answer,
                pas_answer = '',
                student_answer = item_student_answer
            )

            question_context_new = QuestionContext(
                question = question_context.question, 
                string_urls = question_context.string_urls, 
                incorrect_flags = question_context.incorrect_flags,
                marking_note = question_context.marking_note,
                subject = question_context.subject,
                mark_context = mark_context_new,
                answer_context = answer_context_new,
                rubric_context = question_context.rubric_context
            )

            messages = await AutoMarkingChain.get_openai_message_single_answer(
                question_context = question_context_new
            )

            coroutines.append(AutoMarkingChain.get_openai_responses(messages, question_context = question_context_new, debug_mode = debug_mode, auto_fail = auto_fail, api_key = api_key))
        
        full_results = await asyncio.gather(*coroutines)

        return full_results

    @staticmethod
    async def get_result(
        question_context: QuestionContext, 
        reqs: AutoMarking, 
        debug_mode: bool = False,
        auto_fail: bool = False,
        api_key: str = None
    ):
        total_tokens = total_cost = 0
        mark = question_context.mark_context
        models = set()

        # If the marking scheme suggests to do the marking for all blanks at once (for example: "Mark per Question (Admit Partial Marks)")
        if isinstance(mark.fitb_index, list):
            full_results = await AutoMarkingChain.get_result_multi_answer(
                question_context = question_context,
                reqs = reqs,
                debug_mode = debug_mode,
                auto_fail = auto_fail,
                api_key = api_key
            )
            
            for item in full_results:
                total_tokens += item['response'].usage.total_tokens
                total_cost += item['response'].usage.total_cost

            results = [json.loads(item['response'].output_text) for item in full_results]
            models.update([item['model'] for item in full_results])

            messages = await AutoMarkingChain.get_openai_message_multi_answer(
                question_context = question_context,
                marking_results = results
            )

            try:
                response = await AutoMarkingChain.get_openai_responses(messages, type_multi = True, question_context = question_context, debug_mode = debug_mode, auto_fail = auto_fail, api_key = api_key)
                result = json.loads(response['response'].output_text)
                total_tokens += response['response'].usage.total_tokens
            
            except JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Response content: {response['response'].output_text}")
                response = await AutoMarkingChain.get_openai_responses(messages, type_multi = True, question_context = question_context, model = "gpt-4.1", debug_mode = debug_mode, auto_fail = auto_fail, api_key = api_key)
                result = json.loads(response['response'].output_text)

            total_tokens += response['response'].usage.total_tokens
            total_cost += response['response'].usage.total_cost

            models.add(response['model'])

        # If the marking is done per blank only (or for OEQ) 
        else:
            messages = await AutoMarkingChain.get_openai_message_single_answer(
                question_context = question_context,
            )

            try:
                response = await AutoMarkingChain.get_openai_responses(messages, subject = question_context.subject, question_context = question_context, debug_mode = debug_mode, auto_fail = auto_fail, api_key = api_key)
                result = json.loads(response['response'].output_text)
            
            except JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                print(f"Response content: {response['response'].output_text}")
                response = await AutoMarkingChain.get_openai_responses(messages, subject = question_context.subject, question_context = question_context, model = "gpt-4.1", debug_mode = debug_mode, auto_fail = auto_fail, api_key = api_key)
                result = json.loads(response['response'].output_text)

            total_tokens = response['response'].usage.total_tokens
            total_cost = response['response'].usage.total_cost

            models.add(response['model'])

        if debug_mode:
            result['messages_automarking'] = messages
            
        return result, total_tokens, total_cost, models

    ## Automarking Breakdown/Highlight
    async def get_user_prompt_automarking_breakdown(
        messages,
        question_context: QuestionContext,
        marking_result_context: MarkingResultContext
    ):
        """
        Get the user prompt for automarking with multiple answers (Fill in the Blank with multiple blanks and using marking scheme = "Per Question")
        """
        ## Define variables
        question = question_context.question
        language = "english" if not question_context.language else question_context.language.lower()

        answer_context = question_context.answer_context
        correct_answer = answer_context.correct_answer
        student_answer = answer_context.student_answer
        student_image_urls = answer_context.student_image_urls
        
        mark_context = question_context.mark_context
        full_mark = mark_context.full_mark
        fitb_index = mark_context.fitb_index
        num_blanks = mark_context.num_blanks

        ai_mark = marking_result_context.ai_mark
        reason = marking_result_context.reason
        
        vectorstore = False
        type_multi = True if isinstance(fitb_index, list) else False

        if type_multi:
            ## Prerequisites
            instruction_obj = await AutoMarkingChain.get_user_prompt("In this case, there are more than one blank inside the question, each having an answer. Still combine the highlight for each answer into one list under \"mark_highlights\" as well as combining the \"remarks\" using the given AI reasoning (which already combines the feedback).", "text", vectorstore)
            messages.append(instruction_obj)

            ## Question
            question_obj = await AutoMarkingChain.get_user_prompt(f"Here is the composition & question: {question}", "text", vectorstore)
            messages.append(question_obj)
            
            ## Model Answer
            model_answer_prompt = "Here are all the correct answers for all the blanks:"
            for index, answer in enumerate(correct_answer):
                model_answer_prompt += f"\nCorrect Answer for blank {index + 1}: {answer}"
            model_answer_prompt += f"\n\nThe full mark is for the whole question is: {full_mark}."
            model_answer_obj = await AutoMarkingChain.get_user_prompt(model_answer_prompt, "text", vectorstore)
            messages.append(model_answer_obj)

            ## Student Answer
            student_answer_prompt = f"Here are the student's answer for all the blanks:"
            for index, answer in enumerate(student_answer):
                student_answer_prompt += f"\nStudent Answer for blank {index + 1}: {answer}"
            student_answer_prompt += f"\n=============================\nFull Sentence: {await QuestionExtractor().replace_blank(question, student_answer, fitb_index)}"
            student_answer_prompt += f"""\n\nDo not mistakenly alter the student's answer (whether in terms of spelling or others)\n\nAlso, in this case, when returning the "target" for each "mark_highlights" item, ensure you don't return the whole sentence that includes the student answer as the "target". Instead, only highlight the student answer part. This is so that system can do annotation based on exact matching."""
            student_answer_obj = await AutoMarkingChain.get_user_prompt(student_answer_prompt, "text", vectorstore)
            messages.append(student_answer_obj)

            ## AI Mark
            ai_marking_obj = await AutoMarkingChain.get_user_prompt(f"Here is the mark given by the AI automarking system for the student's answer: {ai_mark} | The reason/remarks given by the AI: {reason}", "text", vectorstore)
            messages.append(ai_marking_obj)

            ## Language
            if language.lower() != "english":
                language_obj = await AutoMarkingChain.get_user_prompt(f"For this attempt, the AI has given the reasoning in {language} language. You should do the same, by outputting the rephrased remark (and its highlights breakdown) in {language} language. However, refer to parts of the answers in the original language. You should still refer to parts of the question in their original language. For example, if the question and answer is in English and you are asked to output your reasoning in Vietnamese, you should still use Vietnamese to give your feedback, but refer to parts of the student answer using the original version (english), instead of translating the answer to Vietnamese", "text", vectorstore)
                messages.append(language_obj)
        else:
            ## Question
            question_obj = await AutoMarkingChain.get_user_prompt(f"Here is the composition & question: {question}", "text", vectorstore)
            messages.append(question_obj)
            
            ## Model Answer
            if num_blanks != 0:
                model_answer_prompt = f"Here are the correct answer and alternative answers that are accepted: {correct_answer}. There are {num_blanks} acceptable correct answers. If it matches one of the correct answers, we can consider it as correct. The full mark is {full_mark}."
            else:
                model_answer_prompt = f"Here is the ground truth/model answer: {correct_answer}. The full mark for this question is {full_mark}."
            model_answer_obj = await AutoMarkingChain.get_user_prompt(model_answer_prompt, "text", vectorstore)
            messages.append(model_answer_obj)

            ## Student Answer
            if student_image_urls:
                if isinstance(student_image_urls, str):
                    student_image_urls = [student_image_urls]
                
                student_answer_prompt = await AutoMarkingChain.get_user_prompt(f"The student has submitted the following image(s) as their answer. Please evaluate the content of the image(s):", "text", vectorstore)
                messages.append(student_answer_prompt)

                for img_url in student_image_urls:
                    # Append each image URL as an image_url type prompt
                    messages.append(await AutoMarkingChain.get_user_prompt(img_url, "image_url", vectorstore))
            else:
                student_answer_obj = await AutoMarkingChain.get_user_prompt(f"Here is the student's answer: {student_answer}", "text", vectorstore)
                messages.append(student_answer_obj)

                ## The answer is transcribed from a photo of the whole page, so it
                ## may contain the question/instructions printed on it. Those are
                ## not the student's writing: do not highlight or rephrase them,
                ## only the student's own response.
                if question:
                    exclude_question_obj = await AutoMarkingChain.get_user_prompt("Note: the student's answer above was transcribed from a photo of the page and may include the question, instructions, title, or numbering that was on the page. That text is NOT part of the student's response. Do not create a highlight entry for it and do not rephrase it -- only produce highlights and remarks for the student's own answer to the question.", "text", vectorstore)
                    messages.append(exclude_question_obj)

            ## AI Mark
            ai_marking_obj = await AutoMarkingChain.get_user_prompt(f"Here is the mark given by the AI automarking system for the student's answer: {ai_mark} | The reason/remarks given by the AI: {reason}", "text", vectorstore)
            messages.append(ai_marking_obj)

            ## Language
            if language.lower() != "english":
                language_obj = await AutoMarkingChain.get_user_prompt(f"For this attempt, the AI has given the reasoning in {language} language. You should do the same, by outputting the rephrased remark (and its highlights breakdown) in {language} language. However, refer to parts of the answers in the original language. You should still refer to parts of the question in their original language. For example, if the question and answer is in English and you are asked to output your reasoning in Vietnamese, you should still use Vietnamese to give your feedback, but refer to parts of the student answer using the original version (english), instead of translating the answer to Vietnamese", "text", vectorstore)
                messages.append(language_obj)

        return messages

    async def get_automarking_breakdown(question_context: QuestionContext, marking_result_context: MarkingResultContext, debug_mode, auto_fail, api_key):
        client = DynamicAsyncOpenAI(api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
        model = "gpt-4.1-mini"
        messages = []
        
        ## version 5 (add remarks_simplified)
        ## 260409
        system_prompt = """You are an assistant tasked with analyzing a student's answer to a question and the associated mark and feedback from an AI automarking system. Your objectives are to highlight which parts of the student's answer are correct, partially correct, or incorrect, and to rephrase the developer-oriented AI feedback into clear, student-friendly remarks.

Always follow these rules and steps to ensure clarity, helpfulness, and strict adherence to user-facing feedback guidelines:

# Task Steps

1. **Rephrase AI Feedback:**  
Rephrase the automarking system's remark into friendly, clear, and constructive student feedback. Your rephrasing must:
- Never mention “the AI feedback says/suggests…” or similar phrasing.
- Address the student directly, using language suitable for a learner.
- Provide concept explanations relevant to the question.
- Avoid mentioning technical answer-matching, e.g., "matches one of the acceptable answers.", "similar to the correct answer", etc.
- If the reasoning comes from a rubric, avoid mentioning "According to the rubric,". Instead, you can go straight to the points without mentioning about the word 'rubric'.
- For Sentence Transformation tasks, never use the word "blank"; replace with "question" or "sentence to be filled", referencing position if relevant.
- To ensure the feedback is useful for the students, ensure that the feedback is not just simply about "because your answer is different from the correct answer" if the student's answer is incorrect. This does not explain anything. Instead, touch upon the concept on WHY it is incorrect, and what to do to fix it. For example, if the reasoning behind the mistake is that "using different word changes the meaning", instead of just saying "since it changes the meaning", you can explain as well why it changes the meaning.

2. **Segment the Student Answer:**  
Break down the student's response into discrete sentences. Each sentence should be treated as a separate part for evaluation.

3. **Reason About Marking:**  
Examine the mark and AI feedback, and for each sentence/part, determine its status and provide reasoning:
- If the answer is fully correct (full mark):  
    - Mark all parts as "correct".
    - Give an individual remark explaining why each is correct, referencing relevant concepts.
- If the answer is wholly incorrect (0 mark):  
    - Mark all parts as "incorrect".
    - For each, specify what (word, phrase, or whole sentence) makes it incorrect, and offer conceptual explanation.
- If partially correct (half mark):  
    - If the feedback cites lack of detail/incompleteness, combine all sentences in a single item, mark as "partially_correct", and explain what's missing.
    - If the feedback cites a specific issue (e.g., wrong word, grammar), highlight only the affected sentence/part as "partially_correct" and the others as "correct".
- Do not reference or create highlight entries for missing (unwritten) parts; instead, address omissions in the relevant sentence's remark.

4. **Highlight All Actual Sentences:**  
For every sentence present in the answer, provide a highlight entry—never omit or alter the original sentence text when returning the "target" key.

5. **Compose Each Remark:**  
For each highlight (sentence/part), provide a remark as follows:
- Begin with a brief statement about the status (correct/incorrect/partially correct).
- Justify the classification (why correct/incorrect/partial).
- Provide relevant conceptual explanation.
- End with words of praise/motivation tailored to the answer’s correctness.

6. **Persistence & Completeness:**  
Think through each step. Continue until every objective in the prompt is fully met before producing your final output.

# Notes

- Always rephrase feedback for students; never copy developer jargon.
- Never add highlight entries for sentences not present in the student’s answer.
- Every actual sentence/answer part must get a status and detailed conceptual remark.
- Use the exact sentence from the student answer as the "target" without modification.
- If partial credit is due to a missing detail, mention what's missing in the remark for the relevant written sentence.
- When generating the overall remark along with the detailed remark for each highlighted sentence, ensure that any details that are being put in the highlighted remark is also being put in the overall remark. The objective of this is so the user can generally get the idea from the overall remark.
- End every final output with a JSON object as specified above.
- If the task is about sentence transformation/synthesis and transformation,
    - if the student's answer is correct, you can provide explanation by saying that the student shows a good understanding of how to convert the direct questions into indirect speech.
    - for example, the original sentence is "Lina wondered, "why didn't she return the book on time?" and the student's transformed sentence is "Lina wondered why she had not returned the book on time"
    - since the main objective is to transform the question, the reasoning should not be "since you explained the correct reason why she didn't return the book on time", since its not the objective for sentence transformation questions.
- If the task is related to math and the developer-oriented AI feedback or the student answer contains mathematical expressions in LaTeX format, ensure that you don't manipulate and ensure the equations are wrapped in $$..$$ format. ALWAYS wrap equations in $$ tags only. DO NOT under any circumstances give parentheses or square brackets. For example, you can wrap "\\frac{6}{5}" to become "$$\\frac{6}{5}$$". And don't transform or use ASCII control codes. 
- IMPORTANT: If the question is related to English especially for Grammar related tasks (e.g. Synthesis and Transformation or Sentence Transformation questions), you should not entirely rely on the AI reasoning/remark given to you for you to rephrase. This means, you should also add in more explanations that would be helpful for students to look into. This is done by touching upon the concept of the question more if the student made a mistake in the answer. The concept explanation here will be about on how to fix the mistake. For example, if the student made a mistake in using "this" instead of "that" in reported speech, if its not mentioned in the AI reasoning given to you, you can still add it in your rephrased reasoning, by saying "this" should be changed to "that" to maintain the correct perspective.
- For example, when dealing with a sentence transformation question which requires you to use the "not only... but also..." structure, the ideal concept would be to should keep the sequence of ideas the same as in the original sentences. This helps maintain the original meaning and flow. For example, if the original question is "The apartment is spacious. The apartment is luxurious" and it is being transformed into "The apartment is not only luxurious, but also spacious", then the second part of it "luxurious, but also spacious" can also be commented by saying that the order of the adjectives should match the order of the original sentences.
# Output Format

Output should be a JSON object with:
- `remarks`: the student-friendly, rephrased feedback (string)
- `remarks_simplified`: similar to `remarks` but more concise and simplified for easier understanding (string)
- `mark_highlights`: a list of objects, one per sentence/part in the answer, with:
    - `target`: the original sentence/phrase from the answer (do not alter)
    - `status`: one of "correct", "partially_correct", or "incorrect"
    - `remark`: your reasoning and conceptual explanation for that target

# Example

### Example Input
- Student answer: "Ice floats on water. This is because its density is lower than liquid water."
- AI mark: zero mark
- AI automarking remark: "Matches lemma key. According to the rubric, correct physical principle and clear justification."

### Example Output
{
    "remarks": "Your answer is fully correct. You identified that ice floats because its density is lower than liquid water, which is the essential physical principle behind buoyancy. Great job explaining the science clearly!",
    "remarks_simplified": "Your answer is correct. Ice floats because its density is lower than liquid water.",
    "mark_highlights": [
        {
        "target": "Ice floats on water.",
        "status": "correct",
        "remark": "This statement is correct because it accurately describes the observed phenomenon: ice does indeed float on water."
        },
        {
        "target": "This is because its density is lower than liquid water.",
        "status": "correct",
        "remark": "You correctly explain that ice's lower density compared to liquid water causes it to float, showing good understanding of density and buoyancy."
        }
    ]
}

### Example Input 2 (Math problem)
- Student answer: "\\(79^{\\circ}\\)"
- AI mark: full mark
- AI automarking remark: "The correct answer for \\(\\angle QPS\\) is \\(78^{\\circ}\\) as per the ground truth. The student's answer of \\(79^{\\circ}\\) is close but incorrect. Since the full mark is 2.0 and the answer is not correct or contextually equivalent, no marks can be awarded."

### Example Output 2
{
    "remarks": "Your answer is slightly off from the correct value. The angle $$\\angle QPS$$ is actually $$78^{\\circ}$$, which is the precise measure based on the geometric relationships in the figure. Keep practicing to improve your accuracy in calculating angles!",
    "remarks_simplified": "Your answer is slightly off. The correct angle is $$78^{\\circ}$$.",
    "mark_highlights": [
        {
        "target": "$$79^{\\circ}$$",
        "status": "incorrect",
        "remark": "This answer is close but not quite correct. The exact measure of $$\\angle QPS$$ is $$78^{\\circ}$$, so it's important to pay attention to the precise calculations in geometry problems."
        }
    ]
}

### Example Input 3 (to showcase English Sentence Transformation questions)
- Student answers: "The apartment is" and "luxurious, it is spacious"
- AI mark: zero mark
- AI automarking remark: "For blank 1, the answer 'The apartment is' is correct and matches the ground truth, with no errors. For blank 2, the answer 'luxurious, it is spacious' does not match the required transformation 'spacious but also luxurious'. The student's answer changes the structure and meaning by splitting the qualities into two clauses and using a comma, rather than the required 'not only X but also Y' structure. This results in missing words that should be transformed from the original sentence, which according to the rubric, results in a zero mark for the whole question. Since the rubric states that missing required transformed words or changing the meaning/structure results in zero marks for the entire question, the final mark is 0."

### Example Output 3
{
    "remarks": "Your answer correctly uses 'The apartment is' for the first part, which is exactly what is needed. However, the second part, 'luxurious, it is spacious,' does not follow the required structure of 'not only... but also...'. This structure is important because it links the two qualities in a specific way to show that the apartment has both features together. Instead of separating the qualities into two clauses with a comma, you should keep them connected in one phrase using 'spacious but also luxurious' to maintain the original meaning and flow. Also, remember that when using the structure "not only... but also...", the order of the adjectives should match the order in the original sentences. The phrase should highlight both qualities, but the first quality after "not only" should be the one mentioned first in the original sentences ("spacious"), followed by the second quality ("luxurious") after "but also.". Keep practicing how to use 'not only... but also...' to combine ideas smoothly and accurately.",
    "remarks_simplified": "Your answer is partially correct. The first part is correct, but the second part does not follow the 'not only... but also...' structure.",
    "mark_highlights": [
        {
            'target': 'The apartment is',
            'status': 'correct',
            'remark': 'This part is correct because it matches the required phrase exactly and sets up the sentence properly for the transformation.'
        },
        {
            'target': 'luxurious, it is spacious',
            'status': 'incorrect',
            'remark': "This part is incorrect because it breaks the required 'not only... but also...' structure by splitting the qualities into two separate clauses with a comma. The order of the adjectives should also match the order in the original sentences, to preserve the meaning and grammatical structure."
        }
    ]
}

(For real-world cases, ensure more complex or varied examples as appropriate.)

# Reminder
Review all output to verify reasoning precedes summary/conclusion in every case. Never swap conclusion and reasoning order. Remain concise but detailed. Provide a complete output JSON in all cases.
"""

        system_prompt_object = {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        messages.append(system_prompt_object)

        messages = await AutoMarkingChain.get_user_prompt_automarking_breakdown(
            messages = messages,
            question_context = question_context,
            marking_result_context = marking_result_context
        )

        try:
            response = await client.chat.completions.create(
                model = model,
                messages = messages,
                # temperature = 0,
                # max_tokens = 4095,
                # top_p = 0.1,
                extra_body = {"auto_fail": auto_fail},
                response_format = {
                    "type": "json_object"
                }
            )
        except Exception as e:
            if auto_fail:
                raise e
            if "quota" in str(e).lower():
                print(f"Quota issue. Please check. Using DeepSeek now...")
                client = DynamicAsyncOpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
                response = await client.chat.completions.create(
                    model = "deepseek-v4-flash",
                    messages = messages,
                    # temperature = 0,
                    # max_tokens = 4095,
                    # top_p = 0.1,
                    extra_body = {"thinking": {"type": "disabled"}},
                    response_format = {
                        "type": "json_object"
                    }
                )

        result = json.loads(response.output_text)
        total_tokens = response.usage.total_tokens
        total_cost = response.usage.total_cost
        model = response.model

        ## to filter prerequisites (the AI might not be consistent at outputting $$..$$ l)
        async def filter_prerequisites(text):
            math_regions = []
            for m in re.finditer(r'(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])', text):
                math_regions.append((m.start(), m.end()))

            def is_in_math(pos):
                return any(start <= pos < end for start, end in math_regions)

            # Replace only outside math regions
            def replacer(match):
                start = match.start()
                if is_in_math(start):
                    return match.group(0)
                else:
                    return f"${match.group(0)}$"

            pattern = r'\\frac\{[^}]+\}\{[^}]+\}'
            result = re.sub(pattern, replacer, text)

            ## replace several synbols to ensure consistency
            result = result.replace("\text{o}", "\\circ")
            result = result.replace("\\text{o}", "\\circ")
            result = result.replace("$$$", "$$")

            return result

        pattern = r"\$\$(.*?)\$\$"
        replacement = r"\\(\1\\)"
        pattern_2 = r'\\\\'
        replacement_2 = r'\\'
        pattern_3 = r'\$(.+?)\$'
        replacement_3 = r'\\(\1\\)'

        if result:
            ## prerequisite for fractions and symbols
            result['remarks'] = await filter_prerequisites(result['remarks'])
            result['remarks'] = re.sub(pattern, replacement, result['remarks'])
            result['remarks'] = re.sub(pattern_2, replacement_2, result['remarks'])
            result['remarks'] = re.sub(pattern_3, replacement_3, result['remarks'])

            result['remarks_simplified'] = await filter_prerequisites(result['remarks_simplified'])
            result['remarks_simplified'] = re.sub(pattern, replacement, result['remarks_simplified'])
            result['remarks_simplified'] = re.sub(pattern_2, replacement_2, result['remarks_simplified'])
            result['remarks_simplified'] = re.sub(pattern_3, replacement_3, result['remarks_simplified'])
            
            for index, item in enumerate(result['mark_highlights']):
                print(f"Original target: {item['target']}")

                ## prerequisite for fractions
                result['mark_highlights'][index]['target'] = await filter_prerequisites(item['target'])
                result['mark_highlights'][index]['remark'] = await filter_prerequisites(item['remark'])

                ## clean up
                result['mark_highlights'][index]['target'] = re.sub(pattern, replacement, item['target'])
                result['mark_highlights'][index]['target'] = re.sub(pattern_2, replacement_2, result['mark_highlights'][index]['target'])
                result['mark_highlights'][index]['target'] = re.sub(pattern_3, replacement_3, result['mark_highlights'][index]['target'])
                result['mark_highlights'][index]['remark'] = re.sub(pattern, replacement, item['remark'])
                result['mark_highlights'][index]['remark'] = re.sub(pattern_2, replacement_2, result['mark_highlights'][index]['remark'])
                result['mark_highlights'][index]['remark'] = re.sub(pattern_3, replacement_3, result['mark_highlights'][index]['remark'])

        if debug_mode:
            result['messages_breakdown'] =  messages
            
        return result, total_tokens, total_cost, model

    ## KB Marking
    async def convert_automarking_answers_json(response, api_key):
        client = AsyncOpenAI(api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
        system_prompt = """Your purpose is to convert provided response string from an AI into a JSON instance, by following these rules to convert the provided response string:
- The response string will contain a list of answers generated by an AI. It can be a numbered list, bulleted list, or just a plain text.
- Put all of the items inside "possible_answers", which is a list of items. Do not miss any item.
- Do not put citation mark that might be provided by the AI response into your JSON instance output.

Convert the provided response string into a JSON instance that conforms to the JSON schema: 
{
'possible_answers': ["answer 1", "answer 2", "answer 3", ...]
}"""

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
        question_obj = {"role": "user", "content": [{"type": "text", "text": f"Here is the response string given by the AI:\n{response}"}]}
        messages.append(system_prompt_object)
        messages.append(question_obj)

        response_2 = await client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = messages,
            # temperature = 0,
            # max_tokens = 16384,
            # top_p = 0,
            response_format = {
                "type": "json_object"
            }
        )

        result = json.loads(response_2.choices[0].message.content)
        total_tokens = response_2.usage.total_tokens

        # GPT-4o mini
        total_cost = (response_2.usage.prompt_tokens * 0.15 + response_2.usage.completion_tokens * 0.6)/1e6
        return result, total_tokens, total_cost

    async def get_automarking_answers_assistant(question, vectorstore_id, classification, api_key):
        client = AsyncOpenAI(api_key = api_key or config.OPENAI_API_KEY_DICT["AI_AUTOMARKING"])
        total_cost = 0
        total_tokens = 0
        messages = []
        assistant_id = "asst_1z46sQ6fXrivrKwoEaFvXF5E" ## heyjen temp

        question_prompt = f"Here is the composition & question: {question}"
        user_prompt_obj = {"role": "user", "content": question_prompt, "attachments": [{"file_id": vectorstore_id, "tools": [{"type": "file_search"}]}]}
        messages.append(user_prompt_obj)
        if classification:
            classification_prompt = f"Here is the document type: {classification}"
            classification_prompt_obj = {"role": "user", "content": classification_prompt}
            messages.append(classification_prompt_obj)

        thread = await client.beta.threads.create( 
            messages = messages
        )

        print("thread id:")
        print(thread.id)

        run = await client.beta.threads.runs.create_and_poll(
            thread_id = thread.id, assistant_id = assistant_id
        )

        # GPT-4o
        total_cost += (run.usage.prompt_tokens * 2.5 + run.usage.completion_tokens * 10)/1e6
        total_tokens += run.usage.total_tokens
        messages = list(await client.beta.threads.messages.list(thread_id = thread.id, run_id = run.id))
        response = messages[0][1][0].content[0].text.value

        return response, total_cost, total_tokens, thread.id

    async def get_automarking_answers_alt(question, vectorstore_id, classification, api_key):
        client = AsyncOpenAI(api_key = api_key or config.OPENAI_API_KEY_DICT['AI_AUTOMARKING'])
        system_prompt = """You are an AI that is capable of extracting possible answers from a question. To give you a better understanding of your task, here is the background for you to understand:
1. There is a system called AI automarking that can mark student's answer on a question based on the given correct answer (which can be one or more than one)
2. Each question is stored in a database, manually created by human by manually inputting the question text, correct answers, and solutions.
3. This task can be time consuming, and need another system that can read the same question and then generate the possible correct answers automatically.

To do this, you will be given:
1. The question text, along with the image (if there are any)
2. The knowledge base, attached as a file into the user prompt

Then, your task is to search for relevant informations that can answer the question. Then, you need to output all the possible answers inside a JSON list in the following format:

{
"possible_answers": ["answer 1", "answer 2", ...]
}

When generating your answer, ensure that it is in the same language as in the question's language. For example, if question is in Malay, generate the answers in Malay. If in English, generate in English."""
        
        messages = [{"role": "system", "content": [{"type": "input_text", "text": system_prompt}]}]

        if vectorstore_id:
            messages.append({"role": "user", "content": [{"type": "input_file", "file_id": vectorstore_id}]})

        question_prompt_obj = {"role": "user", "content": [{"type": "input_text", "text": f"Here is the composition & question: {question}"}]}
        messages.append(question_prompt_obj)
        
        if classification:
            classification_prompt_obj = {"role": "user", "content": [{"type": "input_text", "text": f"Here is the document type: {classification}"}]}
            messages.append(classification_prompt_obj)

        run = await client.responses.create(
            model = "gpt-4o",
            input = messages,
            text = {
                "format": {
                    "type": "text"
                }
            },
            # max_output_tokens = 8092,
            # temperature = 0,
            # top_p = 1
        )

        response = run.output[0].content[0].text
        total_tokens = run.usage.total_tokens
        total_cost = (run.usage.input_tokens * 2.5 + run.usage.output_tokens * 10)/1e6
        thread_id = ""

        return response, total_cost, total_tokens, thread_id

    async def get_automarking_answers(question, vectorstore_id, classification, api_key):
        total_cost = 0
        total_tokens = 0

        try:
            response, total_cost, total_tokens, thread_id = await AutoMarkingChain.get_automarking_answers_assistant(question, vectorstore_id, classification, api_key)
        except:
            try:
                print(f"Using Response API")
                response, total_cost, total_tokens, thread_id = await AutoMarkingChain.get_automarking_answers_alt(question, vectorstore_id, classification, api_key)
            except Exception as e:
                print(f"Response API failed after multiple attempts.\nError details: {e}")
                raise Exception("Response API failed after multiple attempts.") from e

        response_2, total_tokens_json, total_cost_json = await AutoMarkingChain.convert_automarking_answers_json(response = response, api_key = api_key)
        total_tokens += total_tokens_json
        total_cost += total_cost_json

        return response_2, total_tokens, total_cost
