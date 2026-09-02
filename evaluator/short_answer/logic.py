import unicodedata

import jiwer
from pydantic import BaseModel, field_validator

from fastapi.responses import JSONResponse
from fastapi import APIRouter, Request, HTTPException

from typing import List, Dict, Union, Literal

import os, json
import re
from bs4 import BeautifulSoup

from openai import OpenAI, AsyncOpenAI
import time
import asyncio

from decimal import Decimal
from typing import Any, Tuple, Optional

STUDENT_UPLOAD_BASE_URL = "https://static-contents-smartjen.s3.ap-southeast-1.amazonaws.com/img/studentUpload/"

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

class AutoMarking(BaseModel):
    question: str
    student_answer: Union[str, List[str]]
    full_mark: float
    step: float = 0.5
    fitb_index: Union[int, List[int]] = 0
    rubric: str = ""
    answer_pas: Union[str, List[str]] = ""
    incorrect_flags: List[Any] = []
    marking_note: str = ""
    marking_schema: str = ""
    vectorstore_id: str = ""
    debug_mode: bool = False
    subject: str = ""
    language: str = "English"
    auto_fail: bool = False

    @field_validator("incorrect_flags", mode="before")
    @classmethod
    def empty_string_to_list(cls, v):
        if v == "" or v is None:
            return []
        if isinstance(v, str):
            try:
                res = json.loads(v)
                if isinstance(res, list):
                    return res
                else:
                    return []
            except Exception:
                return []
        return v
    
    @field_validator("marking_note", "subject", "language", mode="before")
    @classmethod
    def empty_string_to_string(cls, v):
        if v == "" or v is None:
            return ""
        return v

class MarkAllocation(BaseModel):
    question: str
    step: float = 0.5

class GenerateKBAnswer(BaseModel):
    question: str
    file_id: str
    classification: str = ""

## Object Class
class RubricContext:
    def __init__(
            self,
            rubric: str =  "",
            rubric_json: Dict[str, Any] = None,
            string_rubric: str = ""
    ) -> None:
        self.rubric = rubric
        self.rubric_json = rubric_json
        self.string_rubric = string_rubric
    def to_dict(self):
        return {
            "rubric": self.rubric,
            "rubric_json": self.rubric_json,
            "string_rubric": self.string_rubric
        }

class AnswerContext:
    def __init__(
        self, 
        correct_answer: str,
        original_correct_answer: str,
        answer_options: Union[str, List[str]] = None,
        pas_answer: Union[str, List[str]] = "",
        student_answer: Union[str, List[str]] = "",
        student_image_urls: Optional[Union[str, List[str]]] = None,
        solution: Union[str, List[str]] = ""
    ) -> None:
        self.correct_answer = correct_answer
        self.original_correct_answer = original_correct_answer
        self.answer_options = answer_options
        self.pas_answer = pas_answer
        self.student_answer = student_answer
        self.student_image_urls = student_image_urls
        self.solution = solution
    def to_dict(self):
        return {
            "correct_answer": self.correct_answer,
            "original_correct_answer": self.original_correct_answer,
            "answer_options": self.answer_options,
            "pas_answer": self.pas_answer,
            "student_answer": self.student_answer,
            "student_image_urls": self.student_image_urls,
            "solution": self.solution
        }


class QuestionMarkContext:
    """
    Class for Question Mark Context
    -- full_mark: the full mark of the question
    -- step: the increment used for marking (default is 0.5 to allow for partial marks)
    -- type_fitb: boolean indicating if the question is Fill-in-the-Blank type
    -- fitb_index: index or list of indices for FITB questions (default is 0 if non FITB)
    -- num_blanks: number of blanks in the FITB question (if any)
    """
    def __init__(
        self,
        full_mark: Union[int, float] = 0,
        step: Union[int, float] = 0.5,
        type_fitb: bool = False,
        fitb_index: Union[int, List[int]] = 0,
        num_blanks: int = 0,
        unique_blanks: Union[int, List[int]] = 0
    ) -> None:
        self.full_mark = full_mark
        self.step = step
        self.type_fitb = type_fitb
        self.fitb_index = fitb_index
        self.num_blanks = num_blanks
        self.unique_blanks = unique_blanks
    def to_dict(self):
        return {
            "full_mark": self.full_mark,
            "step": self.step,
            "type_fitb": self.type_fitb,
            "fitb_index": self.fitb_index,
            "num_blanks": self.num_blanks,
            "unique_blanks": self.unique_blanks
        }

class QuestionContext:
    def __init__(
        self, 
        question: str = "", 
        string_urls: List[str] = [],
        incorrect_flags: List[Any] = [], 
        marking_note: str = "",
        subject: str = "",
        language: str = "",
        mark_context: QuestionMarkContext = None,
        answer_context: AnswerContext = None,
        rubric_context: RubricContext = None,
    ) -> None:
        self.question = question
        self.string_urls = string_urls
        self.incorrect_flags = incorrect_flags
        self.marking_note = marking_note
        self.subject = subject
        self.language = language
        self.mark_context = mark_context
        self.answer_context = answer_context
        self.rubric_context = rubric_context
    
    def to_dict(self):
        return {
            "question": self.question,
            "string_urls": self.string_urls,
            "incorrect_flags": self.incorrect_flags,
            "marking_note": self.marking_note,
            "subject": self.subject,
            "language": self.language,
            "mark_context": self.mark_context.to_dict(),
            "answer_context": self.answer_context.to_dict(),
            "rubric_context": self.rubric_context.to_dict(),
        }

class AutomarkingSystemPromptConstructor:
    """
    Class to construct system prompt for automarking
    """

    def __init__(self, question_context: QuestionContext):
        self.question_context = question_context
        self.vectorstore = False

        ## define the rest of the parameters
        self.question = question_context.question
        self.string_urls = question_context.string_urls
        self.subject = question_context.subject
        self.language = question_context.language
        self.incorrect_flags = question_context.incorrect_flags
        self.marking_note = question_context.marking_note
        
        self.correct_answer = question_context.answer_context.correct_answer
        self.pas_answer = question_context.answer_context.pas_answer
        self.student_answer = question_context.answer_context.student_answer
        self.student_image_urls = question_context.answer_context.student_image_urls
        
        self.type_fitb = question_context.mark_context.type_fitb
        self.full_mark = question_context.mark_context.full_mark
        self.fitb_index = question_context.mark_context.fitb_index
        self.num_blanks = question_context.mark_context.num_blanks
        self.unique_blanks = question_context.mark_context.unique_blanks
        self.step = question_context.mark_context.step

        self.rubric = question_context.rubric_context.string_rubric ## use the cleaned string version of the rubric, instead of the JSON object version

    async def _construct_general_rules_prompt(self):
        """
        General rules are applied for all subjects except for IGCSE Business for the time being
        """
        if not self.type_fitb and self.subject != "IGCSE Business":
            general_rules_prompt = """\nFrom here, there are several use cases & possibilities you need to learn (which we will call as "general rules"):
1. If the Ground Truth answer is "answer 1 / answer 2 / answer 3" and the PAS answer is "answer 1 (1)". It means that all answer 1, 2, and 3 are treated the same as alternative answers. If the student answers "answer 2", it will still be considered as correct. If the answer is in the ground truth answer (especially if its an alternative answer) but not in the PAS answer, it should be okay and should not be penalized.
2. If PAS answer label is the same as the full mark, but the Ground Truth Answer is not separated by the / symbol, we can still switch to "Partial scores" and consider partial scores. Meaning the maximum value is still the same as the label. For example, the ground truth answer is "Answer 1" and the PAS answer is "Answer 1 (1)" while the full mark is also 1. Since the ground truth answer is not separated by / , we consider it as "partial score", meaning the highest score we can give is 1, but it has the possibility to give half mark.
3. If Ground Truth Answer has no / symbol that indicates alternative answers, any additional informations that are in the ground truth answer but is not in the PAS answer should be considered as "optional", meaning that if not mentioned, it won't penalize the score, but if mentioned, can also consider it as correct. For example, the ground truth is "Answer 1, answer 2, info 3" and the PAS answer is "Answer 1 (1), Answer 2 (1)" (full mark is 2). If the student answers "Answer 1, Answer 2", that means the student did not mention about info 3. But since info 3 is not considered as alternative answer (since its not separated by / sign), its only considered as "additional information", meaning that not mentioning it won't affect student's score. And since student already mention about answer 1 and 2, and both are labelled as (1), student will get (2) because the full mark is 2.
4. If the Ground Truth Answer is in the form of "Answer 1 / Answer 2\nAnswer 3 / Answer 4", that means Answer 2 is the alternative to Answer 1, and Answer 4 is the alternative to Answer 3. Then, if the PAS answer label is "Answer 1 (1) Answer 3 (1)" or "Answer 2 (1) Answer 4 (1)", that means when comparing the student's answer, you should do it per answer basis. If student answers "Answer 3, Answer 4" or "Answer 3\nAnswer 4, since Answer 4 is the alternative of Answer 3, it should be treated the same and it will be marked as 1, not 2, since student is just repeating the same answer with its alternative which has the same meaning.
5. If the PAS answer is separated by / , has been labelled with half mark, and appears twice (which adds the label into full mark), it means mentioning one of the answer will get you half mark, while mentioning two of them will give full mark. For example: the full mark for the answer is 2, and the PAS answer is "Answer 1 / Answer 2 / Answer 3 / Answer 4. (1) Answer 1 / Answer 2 / Answer 3 / Answer 4. (1)", where you can see that "Answer 1 / Answer 2 / Answer 3 / Answer 4. (1)" appears twice. Then, if student answers "answer 1", "answer 2", "answer 3", or "answer 4" only, they will get 1 mark. But if student adds one more answer (can be either answer 1, 2, 3, or 4) (for example: "answer 1\nanswer 2"), they will now get 2 marks.
6. If the PAS answer is in the form of groups like "answer 1 / answer 2 / answer 3 (1), answer 4/ answer 5 / answer 6 (1), answer 7 / answer 8 (1)", it means that there are three groups of answers. To get full mark, student has to mention each item inside each group. The first group of answers (answer 1, 2, and 3) is one group, while the second group of answers (answer 4, 5, and 6) is another group, and the third group (answer 7, 8) is the last group. For this, student has to mention each one of the item from each group to get the full mark. If student answers "answer 1", they will get 1 mark only instead of 3, but if they answer "answer 1\nanswer 4\nanswer 8", they will get 3 mark since they mentioned each item from each groups. But if they answer "answer 1\nanswer 2", they will only get 1 mark since answer 1 & 2 are in the same group, so the answer is repeating. For example, if the PAS answer is "Penguatkuasaan Undang-Undang Darurat / Pemerkasaan Pasukan Keselamatan / Pendaftaran Kebangsaan dan Kad Pengenalan (1) Rancangan Briggs / Tawaran  Pengampunan / Kempen Bulan Rakyat Melawan Penjahat (1) Perintah Berkurung / Perang Saraf / Rundingan Baling (1)" and the student answer is "Rancangan Briggs\nTawaran pengampunan\nPerang saraf", they will get 2 marks instead of 3, since "Rancangan Briggs" is in group 2 while both "Perang Saraf" and "Rundingan Baling" are in the same group 3, so they haven't mentioned any item from group 1 yet.
7. However, if the PAS answer is "answer 1 / answer 2 / answer 3 / answer 4 / answer 5 / answer 6 (3)" and the question specifically mentions to answer 3 items (or if its labelled by the mark such as (3m)), that means each item mentioned in the list will add 1 mark to the score, up to 3. But still, no duplicates are allowed.
8. When the question instruction asks you to specifically answer in one word only, you should strictly follow that rule. This means, if the student answers in more than one word, even if the answer is contextually correct, you should give 0 mark.
9. iF the Ground Truth answer specifically mentions "Accept any TWO:\nAnswer 1\nAnswer 2\nAnswer 3\nAnswer 4\n..", that means as long as the student answers any two of the answers, they will get full mark. If the full mark is 2, then one answer correct awards 1 mark, not 0.5 mark.
"""
        else: 
            general_rules_prompt = ""

        return general_rules_prompt
    
    async def _construct_special_parameter_prompt(self):
        """
        Special parameters are:
        - Incorrect Flags
        - Marking Note
        """
        incorrect_flags = self.incorrect_flags
        marking_note = self.marking_note
        subject = self.subject
    
        special_parameter_prompt_preface = """\n\nApart from the general marking logic/rules, there are also two special parameters that you need to pay attention to:
1. Incorrect Flags
2. Marking Note

These two parameters are optional. The user will tell you whether they exist or not. 
1. Incorrect Flags: While PAS answer will allocate non-zero marks to either part of the answer or the whole answer, the incorrect flags will indicate part of the answer that is incorrect. For example, the PAS answer is "answer A (0.5) answer B (0.5) answer C (1)" and the incorrect flag is "answer D" where answer D is the variation of answer A with different choice of words, that means if student answers "answer D answer B", they will only get 0.5 from mentioning "answer B". They don't get the other 0.5 since answer D gives 0 mark.
2. Marking Note: On top of the general rules that have been mentioned before, there are also marking notes that user will put for a question specifically. This is to tell you specifically the specific rules that you need to follow when grading the answer. For example, the marking note is "only mark answer B if answer A is correct. if answer A is incorrect, the whole answer is considered as incorrect.", that means, if student answers "answer B" only, they will get 0 mark. If they answer "answer A answer B", they will get 1 mark since answer A is 0.5 mark and answer B is 0.5 mark. If they answer "answer A answer C", they will get 0.5 mark since answer A is correct and is required, while answer C is incorrect even after answer A is correct. There can also be notes like "only mark the answer correctly if the answer contains the correct measurement/unit" for answers that contains measurements/units (like 20 mm, 13 cm, etc.). This means, if student only answers the number, they will get 0 since they didnt specify the unit.
If the marking note has points that are overlapping with the general rules, you should follow the marking note instead of the general rules. This is because the marking note is more specific and is tailored to the question.
This also applies to rubrics given by the user (if there are any). If the marking note has points that are overlapping with the rubrics given by the user (for example, mark deduction about punctuation errors, grammar errors, etc), YOU SHOULD follow the marking note instead of the rubric rules first.
User will tell you whether there is a general rubric to follow or not.
"""

        if incorrect_flags or marking_note:
            special_parameter_prompt = special_parameter_prompt_preface

            ## Special case if the marking note is talking about "typo"
            if "typo" in marking_note.lower():
                special_parameter_prompt += """The marking note can be about anything. It can be about only allowing one-word answers.
1. If the question asks you to answer in only one word, then answers that are not one word will be considered incorrect.
2. If the marking note specifically mentions about the number of typos allowed, then you should count the number of typos made in the student's answer. There are two kinds of typos: letter typo (substitution) and missing letters. And when counting the number of typos, you should combine both kinds of typos.
- For example, if the marking note mentions that student answers with up to 2 letters typos are still allowed, then if the correct answer is "caterpillar" and the student answers "aterpillars", you should still consider it as correct since it only has one letter typo (missing c at the beginning of the word. We don't count the additional s at the end of the word as a typo IF the marking note allows for plural form of the answer since that means "caterpillars" is accepted). 
- If the student answers "catapillar", also consider it as correct since the amount of typo is still 2 (switching e to a, and missing r).
- If there is a typo and its less than 2 typos, then you do not need to consider it as partially correct, and should still give full mark (unless the marking note states otherwise).
- Another example, is if the correct answer is "microorganisms" and the student answers "microrganisme", you should still consider it as correct (mark it as full mark) since it still has 2 typos (missing o, and switching s to e at the end of the word). If plural/singular form is allowed, then the number of typos is also two since we compare it to "microrganism" as well (additional e at the end of the word, and missing o).
- Another example, is if the correct answer is "amphibians" and the student answers "amphiby", you should not consider it as correct (mark it as 0) since it has 1 substitution typo (switching i to y) and 2 missing letters (missing n and s), which makes it a total of 3 typos. Also, even though if plural/singular is allowed, "amphiby" is not an official English word.
- Another example, is if the correct answer is "repel" and the student answers "repels", you should consider it as correct (mark it as full mark) since it only has 1 additional letter typo (additional s at the end of the word). Even though we don't count the typo, "repels" is considered as the plural form of "repel" which is acceptable (if the marking note specifies so)
3. If the marking note specifically mentions about allowing either the singular or plural form of the answer, then if the correct answer is "repel" and the student answers "repels", you should still consider it as correct since it is the plural form, and no need to consider it as changing the meaning.
4. The rules about typo and singular/plural form are not related to each other. This means, if the student answer has two letter typos and also uses plural form when the correct answer is in singular form, then it is still considered as correct (since both conditions are still being met). In this case, if the marking note specifies that singular/plural form is allowed, then only penalize based on the number of typos.

Another set of examples:
1. Correct answer: "nymph/nymphs", student answer: "numphgs". Mark: Full mark. Explanation: 2 typos (switching y to u, and additional g), plural form is allowed.
2. Correct answer: "fungus", student answer: "funguzs". Mark: Full mark. Explanation: 2 typos (switching s to z, and additional s at the end).
3. Correct answer: "compass", student answer: "compadd". Mark: Full mark. Explanation: 2 typos (switching s to d, and switching s to d).
4. Correct answer: "Mammary", student answer: "mamaryr". Mark: Full mark. Explanation: 2 typos (missing m, and additional r at the end).
5. Correct answer: "Magnetic", student answer: "magnetikz". Mark: Full mark. Explanation: 2 typos (switching c to k, and additional z at the end).
6. Correct answer: "die", student answer: "dle". Mark: Full mark. Explanation: 1 typo (switching i to l).
7. Correct answer: "die", student answer: "dye". Mark: full mark. Explanation: 1 typo (switching i to y). Even though "dye" is also an English word, but since it only has 1 typo, we still consider it as correct.
8. Correct answer: "Vessel", student answer: "Veesselss". Mark: Full mark. Explanation: 2 typos (additional e, and additional s at the end).
9. Correct answer: "anther", student answer: "amthwr". Mark: Full mark. Explanation: 2 typos (switching n to m, and switching e to w).
10. Correct answer: "snakes/snake", student answer: "snkaes". Mark: Full mark. Explanation: 2 typos (switching a and k's position).
11. Correct answer: "geothermal", student answer: "geptjermal". Mark: Full mark. Explanation: 2 typos (switching o to p, and switching o to j).
12. Correct answer: "migration/migrating", student answer: "mighatiob". Mark: Full mark. Explanation: 2 typos (switching r to h, and switching n to b).
13. Correct answer: "population", student answer: "poppulationm". Mark: Full mark. Explanation: 2 typos (additional p, and additional m at the end).
14. Correct answer: "population", student answer: "populatonm". Mark: Full mark. Explanation: 2 typos (missing i, and additional m at the end).
15. Correct answer: "elastic", student answer: "elasyiv". Mark: Full mark. Explanation: 2 typos (switching t to y, and switching c to v).

Another set of examples (where the number of typos exceed the allowed limit):
1. Correct answer: "fungus", the student answer: "fugnuss", max allowed typos: 2. Mark: 0. Explanation: 3 typos (switching n to g, g to n, and additional s at the end).
2. Correct answer: "compass", the student answer: "compaddd", max allowed typos: 2. Mark: 0. Explanation: 3 typos (switching s to d, switching s to d, and additional d at the end).
3. Correct answer: "shark/sharks", the student answer: "sharcksx", max allowed typos: 2. Mark: 0. Explanation: 3 typos (switching k to c, additional k, and additional x at the end).
4. Correct answer: "pupae", the student answer: "popeea", max allowed typos: 2. Mark: 0. Explanation: 3 typos (switching u to o, switching a to e, and additional a at the end).
5. Correct answer: "pupae", the student answer: "pooppae", max allowed typos: 2. Mark: 0. Explanation: 3 typos (switching u to o, additional p, and additional o).
6. Correct answer: "like/likes", the student answer: "liekks", max allowed typos: 2. Mark: 0. Explanation: 3 typos (switching e to k, k to e, and additional k).
7. Correct answer: "flexibilty", the student answer: "fleksibiliti", max allowed typos: 2. Mark: 0. Explanation: 3 typos (switching x to k, additional s after k, and changing y to i at the end).
8. Correct answer: "mammary", the student answer: "mmamaryy", max allowed typos: 2. Mark: 0. Explanation: 3 typos (additional m at the beginning, additional y at the end, and missing m in the middle).
"""
                
            ## subject specific rubric
            if "IGCSE Business" in subject:
                special_parameter_prompt = f"""{special_parameter_prompt_preface}

In this case since you will be handling for IGCSE Business/Economics questions, the marking note will mostly be about how to give the mark based on the number of points/complexity of the student's answer. Here is what you need to know:
**Marking for Explain/Outline/Describe questions:**
- Use the [k] (identification), [app] (application), and [an] (explanation/analysis) model **if required by the mark scheme or marking note**.
- **Always check the marking note or ground truth given to you** to see which elements ([k], [app], [an]) are required for full marks for this question.
- Each mark must be awarded for a **distinct, non-overlapping phrase** in the student's answer.
- [k]: The main point, method, or way.
- [app]: Application to the business/context (e.g., mentioning "flowers" or "shop").
- [an]: A **separate, explicit explanation** of how or why the method increases added value (or answers the question).
- **Do not infer explanation from application or identification.** Each must be present as a separate phrase.
- If the explanation ([an]) is only implied or not clearly stated, **do not award the [an] mark**.
- **Do not award the same point twice** (e.g., do not use the same phrase for both [k] and [an]).
- If the question requires you to present the identification first before mentioning the reasoning/application/explanation, then if the student made an incorrect point/identification even with correct application, then do not award any marks for that point.

**Example:**
- If the mark scheme says: "Award 1 mark for identification, 1 mark for application, 1 mark for explanation," then all three are needed.
- If the mark scheme says: "Award 1 mark for each relevant way, 1 mark for each relevant reference to this business," then only [k] and [app] are needed.
- If the mark scheme says: "Award 1 mark for each relevant way," then only [k] is needed.

Example Question 1:
- Question is asking about advantage/disadvantage about PF selling its products in mass market
- Marking note is: "Award 1 mark for each advantage/disadvantage of selling in a mass market (max 2). Award a maximum of 3 additional marks for each explanation of the advantage/disadvantage of selling in a mass market, one of which must be applied to this context."
- The marking note basically means, the total mark is 8, and the student will get 1 point each for mentioning the advantage/disadvantage. Meaning if the student gets both correct, they get 2.
- As for the rest of the 6 points, the student has to further elaborate each advantage/disadvantage. The explanation of the advantage will award the student with 3 mark, and the explanation of the disadvantage will also award 3 mark.
- In total, the student can get up to 8 marks. An explanation does not have to be a paragraph length each, but can be just a simple statement.
-- Student Sample Answer 1:
- For example: Answer for advantage: "Total sales to the market may be very high (1) which is likely to give the potential for high revenue (1) from the high sales of shoes (app). Possibly leading to high profits if costs are kept under control (1).". In this case, it mentions the advantage (total sales may be very high) which gives 1 mark, then it explains the advantage (likely to give potential for high revenue) which gives 1 mark, then it applies to the context/mentioning the application (from high sales of shoes) which gives 1 mark, and finally it further explains the advantage (possibly leading to high profits if costs are kept under control) which gives another 1 mark. So in total, the student will get 4 marks for the advantage part.
- As for the application, it might include: rubber shoes; natural rubber raw material; building a new factory; rubber is grown locally; increase in taxes; bank loan for new factory; sister wants to invest in the business; $100000; operates in a very competitive market; started 10 years ago. Other than these, other applications are not considered as valid application.
-- Student Sample Answer 2:
- The student answers: "One advantage is that total sales may be very high, which can lead to high revenue and profits if costs are controlled. One disadvantage is that there is high competition, so businesses might need to lower prices, reducing profit margins."
- The example above shows that the student was able to show the advantage (sales might be very high) as well as the disadvantage (high competition). So the student will get 2 marks for that.
- Then, the student also explained the advantage (can lead to high revenue, lead to profits if costs are controlled) and also explained the disadvantage (business might need to lower prices, and business need to reduce profit margins). So the student will get 2 marks for the advantage explanation, and 2 marks for the disadvantage explanation.
- In total, the student will get 6 mark (missing another line of explanation or the application for either advantage or disadvantage to get full 8 marks).
-- Student Sample Answer 3:
- Student answers: "Advantage: Lower production costs from standardised products. Disadvantage: High competition forces price cuts.". In this answer, the student only mentions about the advantage/disadvantage, but fail to add an explanation. So the student will only get 2 marks.
-- Student Sample Answer 4:
- Student answers: "PF can sell to more people". In this answer, the student only mentions about the advantage (with no explanation), but fail to mention the disadvantage as well as the explanation. So the student will only get 1 mark. 
-- Student Sample Answer 5:
- Student answers: "PF can grow due to high sales in a mass market. However, they might face high competition, which could lead to lower profit margins.". They will get 3 since they get both advantage and disadvantage (high sales in mass market & might face high competition), which is 2 marks in total, but only explained the disadvantage (which could lead to lower profit margins), which results in 1 mark. So in total, the student will get 3 marks.
-- Student Sample Answer 6:
- Student answers: "PF can achieve high revenue by selling in a mass market. However, they might need to reduce prices to remain competitive.". They will get 2 since they only mention both the advantage as well as the disadvantage (high revenue by selling in a mass market & need to reduce price to remain competitive). However, those points cannot be used as the explanation as well. So in total, the student will get 2 marks.
-- Student Sample Answer 7:
- Student answers: "PF can produce a standardised product, which lowers costs. However, they might lose out to niche market products.", the disadvantage point "lose out to niche market products" cannot be considered as an explanation as well. So in total, the student will only get 3 marks (1 for advantage point, 1 for disadvantage point, and 1 for advantage explanation).
-- Student Sample Answer 8:
- Student answers: "PF can sell more products in a mass market. But they might face challenges.". For advantage, "PF can sell more products in a mass market." is awarded 1 mark since its a correct identification of an advantage (without any explanation or application). However, for the second part, "But they might face challenges.", its a bit too vague and does not specify what kind of challenges. So no marks will be awarded for the disadvantage. In total, the student will get 1 mark.

Example Question 2:
- Question is asking about "Outline two ways an economic boom might affect someone's business"
- Marking note is: "Award 1 mark for each relevant way (max 2). Award 1 mark for each relevant reference to this business (max 2). Other appropriate responses should be credited.
- The marking note basically means, the total mark is 4, and the student will get 1 point each for mentioning the relevant way. Meaning if the student gets both correct, they get 2.
- As for the rest of the 2 points, the student has to further mention the reference/direct application to the business (by mentioning a specific term if required). The reference to the business will award the student with 2 marks.
-- Student Sample Answer 1:
- Student answers: "An economic boom might increase demand for flowers, and it could be harder to find employees for the shop."
- The first point, "An economic boom might increase demand for flowers" mentions the relevant way (increaase demand) as well as the application/relevant reference (for flowers). So the student will get 2 marks for that.
- The second point, "it could be harder to find employees for the shop" mentions the relevant way (harder to find employees) as well as the application/relevant reference (for the shop). So the student will get 2 marks for that.
- In total, the student will get 4 mark.
-- Student Sample Answer 2:
- Student answers: "There will be more demand for flowers and higher costs."
- The first point, "There will be more demand for flowers" mentions the relevant way (increase demand) as well as the application/relevant reference (for flowers). So the student will get 2 marks for that.
- The second point, "higher costs" mentions the relevant way (higher costs), but does not mention the application/relevant reference. So the student will get 1 mark for that.
- In total, the student will get 3 mark.
-- Student Sample Answer 3:
- Student answers: "An economic boom means the government will give Brendan more support and he will have less competition."
- The first point, "the government will give Brendan more support" does not necessarily mention the relevant way, since government support does not necessarily increase because of economic boom, but rather the wages/costs that might increase, which might need government support to help.
- The second point, "he will have less competition" is not necessarily true since in an economic boom, more businesses might open up, which increases competition instead of reducing it.
- In total, the student will get 0 mark.
-- Student Sample Answer 4:
- Student answers "Brendan can increase his prices during an economic boom, which will increase the added value of his flower shop. However, he might also face higher costs as wages increase."
- The first point, "Brendan can increase his prices during an economic boom" mentions the relevant way (increase prices) as well as the application/relevant reference (his flower shop). So the student will get 2 marks for that.
- The second point, "he might also face higher costs as wages increase" mentions the relevant way (higher costs as wages increase) but does not mention the application/relevant reference. So the student will get 1 mark for that.
- In total, the student will get 3 mark.
-- Student Sample Answer 5:
- Student answers "An economic boom will make it easier for Brendan to start his flower shop"
- The point, "An economic boom will make it easier for Brendan to start his flower shop" does not necessarily mention the relevant way, since an economic boom does not necessarily make it easier to start a business. It might even be harder due to higher costs.
- In total, the student will get 0 mark.
-- Student Sample Answer 6:
- Student answers "Breandan could set high prices for his flowers, which would widen the gap between the cost and price, increasing added value. He could also improve the quality of his flowers to make them more desirable."
- The first point, "Breandan could set high prices for his flowers, which would widen the gap between the cost and price" mentions the relevant way (set high prices), the reference/analysis (widen the gap between cost and price), as well as the application/relevant reference (for his flowers). So the student will get 3 marks for that.
- The second point, "He could also improve the quality of his flowers to make them more desirable." mentions the relevant way (improve quality), as well as the application/relevant reference (for his flowers). However, it does not mention the analysis/reference of how improving quality will help increase added value. So the student will get 2 marks for that.
- In total, the student will get 5 mark.
- Student Sample Answer 7:
- Student answers "He could use branding to give the idea of higher quality and improve convenience by making flowers immediately available in the shop."
- The first point, "He could use branding to give the idea of higher quality" mentions the relevant way (use branding), as well as the analysis/reference (give the idea of higher quality). However, it does not mention the application/relevant reference (for his flowers/shop). So the student will get 2 marks for that.
- The second point, "improve convenience by making flowers immediately available in the shop." mentions the relevant way (improve convenience), as well as the application/relevant reference (in the shop). It also mentions the analysis/reference (making flowers immediately available). So the student will get 3 marks for that.
- In total, the student will get 5 mark.

Example Question 3:
- Question is asking about "Explain two ways a businessman could increase added value for his business. 
- Correct answer contains samples that are using [k], [an], and [app], with maximum score of 6 (3 points for each elaborated point)

Student Sample Answer 1:
- Student answers "He could buy cheaper flowers from suppliers to lower his costs. He could also make his shop more convenient for customers by opening early in the morning."
- Mark Scheme Analysis:
First Point: Buy cheaper flowers from suppliers
- **Identification (k):** "buy cheaper flowers from suppliers" (1 mark)
- **Application (app):** "flowers" (applies to the business context) (1 mark)
- **Explanation (an):** "to lower his costs" (explains how this increases added value) (1 mark)
Second Point: Make shop more convenient by opening early
- **Identification (k):** "make his shop more convenient for customers" (1 mark)
- **Application (app):** "by opening early in the morning" (applies to the shop context) (1 mark)
- **Explanation (an):** The answer does not explicitly explain how this increases added value (e.g., by attracting more customers or increasing sales), so the explanation mark is not fully justified.
Total mark: 5 out of 6 marks.
Student Sample Answer 2:
- Student: "He could make his shop more convenient for customers by opening early in the morning."  
- Marking:  
- [k]: "make his shop more convenient for customers" (1 mark)  
- [app]: "by opening early in the morning" (1 mark)  
- [an]: **No mark** (no explicit explanation of how this increases added value)  
- Total: 2 marks for this point, not 3.

Student Sample Answer 2:
- Student answers "He could use a unique design for his flowers"
- This answer only mentions one point along with the application (use unique design [k] for his flowers [app]), but as mentioned above, we can't implicitly assume the explanation (an) is present because it is not mentioned in the answer. So the student will only get 2 marks.

Student Sample Answer 3:
- Student answers "He should just sell more flowers"
- Even though this answer mentions about "flowers", which is the application (app), but it does not specifically mention about the identification (k) of how to increase added value, nor does it explain (an) how selling more flowers will increase added value. So the student will get 0 marks.

Example Question 4:
- Question is asking about "Explain two pricing methods a new business might use. Which is likely to be the best method to use? Justify your answer."
- For this question, if the marking note mentions about "To access evaluation must discuss two relevant methods", that means:
-- In order for the evaluation part to be awarded full (2 marks for example), then both pricing method must be compared/evaluated. If the evaluation is just talking about one method, then the student should get 1 mark.
-- If the student only justifies why one method is best without comparing it to the other method, then the student will get 1 mark instead of 0 for evaluation.

Student Sample Answer 1:
- Student answers "A business could use cost-plus pricing, which is adding a markup to the cost. Another is promotional pricing, where you offer discounts for a short time. Cost-plus is best because it is simple and ensures you cover your costs."
- In this answer, the student mentions both the method [k] as well as the relevant development of each points [an]. However, when it comes to the evaluation part, the student only evaluates one method (cost-plus pricing) without evaluating promotional pricing. So the student will only get 1 mark (not 0 since the student still mentions evaluation of the first method) for the evaluation part.
- In total, the student will get 5 marks (2+2+1).

Student Sample Answer 2:
- Student answers "Price skimming allows a business to recover development costs quickly. Psychological pricing can make prices seem lower. I think psychological pricing is the best because it can influence customer perception."
- In this answer, the student mentions both the method [k] as well as the relevant development of each points [an]. However, when it comes to the evaluation part, the student only justifies why psychological pricing is best without comparing it to price skimming. So the student will get 1 mark (not 0 since the student still mentions justification for one method) for the evaluation part.
- In total, the student will get 5 marks (2+2+1).

Example Question 5:
- Question is asking about "Explain two advantages and two disadvantages to Philip of changing PF from a sole trader business to a private limited company."
- For this question, the student requires to mention each point along with its application in order to get 2 points for each advantage/disadvantage, which in total gives 8 marks.
- In this case, each point does not need to be explained, but needs to be followed by application.
- Application could include: rubber shoes; natural rubber raw material; build a new factory; sells in a mass market; increase in taxes; bank loan for the new factory; sister wants to invest in the business; $100 000; wants to grow the business; produce a new range of shoes; operates in a very competitive market; started 10 years ago;

- Student Sample Answer 1:
- Student answers "Limited liability means Philip's personal assets are protected. More capital can be raised by selling shares. He won't keep all profits anymore. There are legal restrictions to set up."
- In this answer, the student mentions two advantages points [k] and two disadvantages points [k], but only the advantage points are elaborated with application [app] (personal assets are protected; selling shares). So the student will only get 6 marks (2 marks for each advantage, 1 mark for each disadvantage).

- Student Sample Answer 2:
- Student answers "Philip will not be able to keep all the profit made from selling the shoes he produces"
- In this answer, the student only mentions one disadvantage point [k] (not able to keep all profits) but they mention the application of it (from selling the shoes he produces) [app]. So the student will get total of 2 marks for this point.

- Student Sample Answer 3:
- Student answers "Advantage: Limited liability for his $100,000. Disadvantage: Legal restrictions when setting up."
- In this answer, the student mentions one advantage point [k] (limited liability) along with its application [app] ($100,000), and one disadvantage point [k] (legal restrictions) without any application (process of setting up is not considered as the application). So the student will get total of 3 marks (2 marks for advantage, 1 mark for disadvantage).

- Student Sample Answer 4:
- Student answers "Limited liability is an advantage because Philip's personal assets are protected. A disadvantage is that he won't keep all the profits."
- In this answer, the student mentions one advantage point [k] (limited liability) along with its application [app] (personal assets are protected), and one disadvantage point [k] (won't keep all profits) without any application. So the student will get total of 3 marks (2 marks for advantage, 1 mark for disadvantage).

- Student Sample Answer 5:
- Student answers "One advantage is continuity, as PF will still exist if he leaves. Another advantage is that he can raise more capital. A disadvantage is that he won't have complete control."
- In this answer, the student mentions two advantage points [k] (continuity and rase more capital). The first point is also followed by the application [app] (PF will still exist if he leaves). The second point does not have any application. As for the disadvantages, the student only mentions one disadvantage point [k] (won't have complete control) without any application. So the student will get total of 4 marks (2 marks for first advantage, 1 mark for second advantage, 1 mark for disadvantage).

- Student Sample Answer 6:
- Student answers "Philip will have limited liability, which means his personal assets are safe. He can also raise more money by selling shares. However, he will not be ablt to keep all the profits, and he may lose some control over the business."
- In this answer, the student mentions two advantage points [k] (limited liability and raise more money). The first and second point are also followed by the applications [app] (personal assets are safe; selling shares). As for the disadvantages, the student mentions one disadvantage point with application (lose some control over the business) but the other disadvantage point (not able to keep all profits) does not have any application. So the student will get total of 7 marks (2 marks for each advantage, 1 mark for first disadvantage, 2 marks for second disadvantage).

Example Question 6:
- Question is asking the student to analyse the influences on whether demand for shoes is elastic or inelastic based on certain factors.
- In this case, there are no marking notes provided, and the full mark is 6.
- Because of that, do the marking based on each point provided by the user. The example of points will be shown in the provided ground truth answer.

- Student Sample Answer 1:
- Student answers "The demand for shoes depends on whether they are a luxury or a necessity. If they are a necessity, demand will be inelastic. Also, if there are many substitutes like sandals and boots, demand could be elastic.".
- In this answer, there are three points made by the student: "The demand for shoes depends on whether they are a luxury or a necessity" (1 mark), "If they are a necessity, demand will be inelastic" (1 mark), and "if there are many substitutes like sandals and boots, demand could be elastic" (1 mark). So the student will get total of 3 marks.

- Student Sample Answer 2:
- Student answers "Shoes can be elastic or inelastic depending on the situation. For example, if people can wait to buy new shoes, demand is elastic. If they can't wait, it's inelastic. Also, if shoes last a long time, demand is inelastic."
- In this answer, there are four points made by the student: "Shoes can be elastic or inelastic depending on the situation" (1 mark), "if people can wait to buy new shoes, demand is elastic" (1 mark), "If they can't wait, it's inelastic" (1 mark), and "if shoes last a long time, demand is inelastic" (1 mark). So the student will get total of 4 marks.

Example Question 7:
- The question is asking "Analyse the relationship between the literacy rate and the percentage of the labour force employed in agriculture."
- This question has total of 5 marks, and is asking the student to provide five different points: 1) Expected Relationship, 2) Supporting Evidence, 3) Analysis of expected relationship, 4) Exception, 5) Analysis of exception.
- In this case, the note will be: "Responses do not have to be in the format suggested but they should address the expected/normal relationship, offer supporting evidence of that, highlight any exceptions to that, and analyse the overall data. Supporting evidence should involve interpretation, not just description. Simply stating percentages without analysis of being high or low gets no marks. Accept as analysis where two countries with different levels of adult literacy and percentage employed in agriculture are directly compared."

- Student Sample Answer 1:
- Student answers "There is a negative relationship between literacy rate and the percentage of people working in agriculture. South Africa has the highest literacy and lowest agriculture employment. Mali and Sierra Leone have low literacy and high agriculture employment. Gabon is an exception because it has high literacy but also high agriculture employment." 
- In this case, the student mentions the expected relationship ("There is a negative relationship between literacy rate and the percentage of people working in agriculture") (1 mark). For the supporting evidence, they mentioned about the evidence, as well as being able to provide the analysis to the evidence by doing a direct comparison of the literacy rate and agriculture employment in South Africa, Mali and Sierra Leone. Because of that, they get 1 mark for supporting evidence as well as 1 mark for the analysis. For the exception, they mentioned about Gabon that has high literacy and high agriculture employment (1 mark). However, they did not provide any analysis of the exception as well (0 mark). So in total, the student will get 4 marks.

- Student Sample Answer 2:
- Student answers "There is generally a negative relationship between literacy rate and the percentage of the labour force employed in agriculture. For example, South Africa has the highest literacy rate and the lowest percentage in agriculture, while Mali and Sierra Leone have low literacy rates and high percentages in agriculture. This suggests that higher literacy rates allow more people to work in other sectors. However, Gabon is an exception because it has a high literacy rate but also a high percentage of its labour force in agriculture. This could be because other sectors are small or some agricultural jobs require skills."
- In this case, the student mentions the expected relationship ("There is generally a negative relationship between literacy rate and the percentage of the labour force employed in agriculture") (1 mark). For the supporting evidence, they mentioned about the evidence, as well as being able to provide the analysis to the evidence by doing a direct comparison of the literacy rate and agriculture employment in South Africa, Mali and Sierra Leone. Because of that, they get 1 mark for supporting evidence as well as 1 mark for the analysis. For the exception, they mentioned about Gabon that has high literacy and high agriculture employment (1 mark). They also provided analysis of the exception (1 mark). So in total, the student will get full 5 marks.

- Student Sample Answer 3:
- Student answers "There is a negative relationship between literacy rates and agricultural employment. South Africa has the highest literacy rate and the lowest agricultural employment. Gabon is an exception."
- In this case, the student mentions the expected relationship ("There is a negative relationship between literacy rates and agricultural employment") (1 mark). For the supporting evidence, they mentioned about South Africa's literacy rate and agricultural employment, however they did not provide any analysis of the evidence (0 mark) by making a direct comparison with other countries. The marking note specifically mentions that, simply stating percentages without analysis of being high or low will get the point no marks. In this case, both "supporting evidence" and "analysis" are combined into one, meaning if the analysis is 0, then supporting evidence will be 0 as well. That means, both "supporting evidence" and "analysis of expected relationship" are 0. For the exception, they only mentioned about Gabon being an "exception", but they did not elaborate on how is Gabon an exception (by saying that Gabon has high literacy rate but also high agriculture employment for example). In this case, they will get 0 mark for the exception. They also did not provide any analysis of the exception as well (0 mark). So in total, the student will get 1 marks.

- Student Sample Answer 4:
- Student answers "Higher literacy means fewer people work in Agriculture. South Africa is an example. Gabon is different."
- In this case, the student mentions the expected relationship ("Higher literacy means fewer people work in Agriculture") (1 mark). However, for the supporting evidence, they only mentioned South Africa as an example without any analysis of the evidence (0 mark). The marking note specifically mentions that, simply stating percentages without analysis of being high or low will get the point no marks. In this case, both "supporting evidence" and "analysis" are combined into one, meaning if the analysis is 0, then supporting evidence will be 0 as well. That means, both "supporting evidence" and "analysis of expected relationship" are 0. For the exception, they only mentioned Gabon being "different", but they did not elaborate on how is Gabon an exception (by saying that Gabon has high literacy rate but also high agriculture employment for example). In this case, they will get 0 mark for the exception. They also did not provide any analysis of the exception as well (0 mark). So in total, the student will get 1 marks.

Example Question 8:
- The question is asking "Discuss whether or not a government should pay private sector firms to give work experience to unemployed young people."
- The keyword here is "Discuss", in which each point made has to be elaborated with reasoning. 
- In this case, if the full mark is 6, that means the student can either make: 1) 2 points discussing about why the government should pay private sector firms and 1 point on why the government should not, or 2) 2 points discussing about why the government should not pay private sector firms and 1 point on why the government should
- Either way, the student cannot simply provide 3 points from each (should/should not) side without any elaboration/reasoning. Each point must be elaborated with reasoning to get the mark.

- Student Sample Answer 1:
- Student answers "The government should pay private sector firms to give work experience to unemployed young people because it can help them develop skills and increase their chances of getting a job. This could reduce youth unemployment and help the economy grow. However, the quality of the work experience might not be good, and the government would have to spend money that could be used elsewhere."
- The first point, "it can help them develop skills" is a valid point and is developed by giving a reasoning behind the point ("increase their chances of getting a job") (1 mark).
- The second point, "This could reduce youth unemployment" is also a valid point and is also developed by giving a reasoning behind the point ("helps the economy grow") (1 mark). 
- As for the third and fourth point, "However, the quality of the work experience might not be good" and "government would have to spend money that could be used elsewhere" are also both valid points against the idea but without further development from reasoning. So the student will get total of 2 marks (1 for each undeveloped point).
- In total, the student will get 6 marks (4 for developed points for why the government should and 2 for undeveloped points for why the government should not).

- Student Sample Answer 2:
- Student answers "The government should not get involved because it is up to the private sector to hire people."
- The only point is "it is up to the private sector to hire people", which is a valid point against the idea but without further development from reasoning. So the student will get total of 1 mark.

- Student Sample Answer 3:
- Student answers "The government should pay private sector firms to give work experience to unemployed young people because it can help them develop skills and increase their chances of getting a job. This can reduce unemployment and increase tax revenue, leading to economic growth."
- The first point, "it can help them develop skills" is a valid point (1 mark) and is developed by giving a reasoning behind the point ("increase their chances of getting a job") (1 mark).
- The second point, "This can reduce unemployment" is also a valid point (1 mark) and is also developed by giving a reasoning behind the point ("increase tax revenue, leading to economic growth") (1 mark).
- In total, the student will get 4 marks.

- Student Sample Answer 4:
- Student answers "The government should pay firms because it can reduce unemployment and increase tax revenue. However, the young may be given unskilled jobs, which might not be beneficial."
- The first point, "it can reduce unemployment" is a valid point (1 mark) and is developed by giving a reasoning behind the point ("increase tax revenue") (1 mark).
- The second point, "the young may be given unskilled jobs" is also a valid point against the idea (1 mark) and with a reasonable explanation of why it is not beneficial (which is "might not be beneficial") (1 mark).
- In total, the student will get 4 marks.
"""

            if 'IGCSE English' in subject:
                special_parameter_prompt = f"""{special_parameter_prompt_preface}

In this case, if the marking note tells you to give marks by following a certain mark scheme for example:
- 1 mark: relevant inference, 1 mark: reference to word/phrase, 1 mark: clear explanation
If the correct answer does not follow it, then you should prioritize the correct answer first. 
For example, if the question is: "What effect does the informal letter style have on the reader?" and the correct answer is "It makes the writer sound honest and relatable.", then if the student answers "The informal letter style makes the writer sound honest and relatable.", it should still be considered as correct even though it does not follow the marking note structure. You should give full mark for this answer.
"""

        else:
            special_parameter_prompt = ""
        
        return special_parameter_prompt
    
    async def _construct_subject_specific_prompt(self):
        """
        Subject Specific Prompt
        Different subjects might have different instructions
        """

        subject = self.subject

        ## Get Subject Specific Prompt
        if any(s in subject for s in ["English"]):
            subject_specific_prompt = """\nThere are also subject specific rules. For this case, you will be handling questions from English subject.
**English**
For English: Open Ended Questions/general questions:
1. If the question asks you to answer in only one word, then answers that are not one word will be considered incorrect.
2. If an answer from a student is not matching 100% to the ground truth answer or the PAS answer structurally, but is close in terms of context, then the answer should be regarded as correct (give the mark according to the PAS label), which is why context matters. This means, student's answers can be varying and does not need to explicitly state the answer in the PAS answer. If you need to decide whether the answer is contextually close to the ground truth or PAS answer, you need to check the question and composition provided by the user. Then you will know what we can accept.
3. Student answers that are a subset of the ground truth answer, as long as the context is correct, should be regarded as correct also. For example, the question is "Why is the exhibit on frogs ('Frogs Forever?') titled as a question?" and the PAS answer is "It is a question as frogs might disappear \/ go extinct and not be around forever. (1)". Here, you can see that "and not be around forever" is just a complimentary additional information that is added in the question. If student answers "It is a question as frogs might disappear", it already addresses the main focus of the question, which means it should be marked as full mark.
4. Generally, typos/mispellings in website URLs, sites, etc. in the answer are considered as non-damaging typos. As long as the point the student wants to bring up stays the same, URL mistakes should not be penalized. For example, writing "website.co.id" instead of "website.com" should not be penalized, since the difference is only on the domain. But in the scope of science (physics, chemistry, biology), if the typo is crucial (for example, substance names, species, etc.), then it should be penalized.
5. When reading the Ground Truth answer or PAS answer and comparing it to the general rules mentioned above, you also have to understand that: For English, not all / or \/ signs indicate alternative answers. For example, "unaware / non chalant" means the same, meaning it is used as a synonym for that word. So you need to also know whether the / signs are used for alternative answers or for synonyms. The usage of / for synonyms will make the grading most likely to be less generous. Synonyms can be for phrase/word basis, for example: "they are facing a sad/sorry/unfortunate situation" means sad, sorry and unfortunate are interchangable, as the three of them has the same meaning in that context. So, answering "they are facing an unfortunate situation" is considered as correct, but answering "sad" or "unfortunate" only might not be as it lacks further information.
6. Directions like "west", "east", "north", "south", etc are not considered as proper nouns. So in this case, you shouldn't treat them like one, and does not need to be written with a capital letter.

For English: Sentence Transformation or Synthesis and Transformation questions:
- General rules (if no specific rubric present):
1. There are going to be cases where there are multiple answer possibilities in the ground truth but only one of the item is inside the PAS answer. In order for the answer to be marked as full mark, the answer does not have to be exactly the same as one of the alternative answers or the same as the PAS answer. For example, in Sentence Transformation questions, different choice of words that are not used in the ground truth answer or PAS answer can still be accepted as long as the meaning does not change. 
2. Any mistakes in punctuation should cause the answer to be 0. For example: ending the answer in a comma instead of a period. However, not ending the answer with a period is fine and should not be penalized, as long as it is not ended with a comma.
3. Different choice of wordings will not cause in score deduction. For example, the question asks to transform the sentence "The baby is cute. Everyone adores her" the PAS answer is "adores the baby's cuteness" and the ground truth answer is "adores the baby because she is cute". If the student answer is "adores the baby as she is cute" or "adores the baby since she is cute", it will still be considered as correct since "because" is similar to "as" and "since" in this context and does not change the meaning.
4. However, if it's not different choice of wordings but missing a word instead, this should be highlighted and be explained in your reasoning. An example of missing word is: The correct answer is "Unless it stops raining, we are not going to John’s house today" but the student answers "Unless it stops raining, we are not going to John's house", missing the word "today", which is crucial for Sentence Transformation questions and affects the meaning.
5. When comparing for grammar mistakes, please be mindful and smart. For example, having your feedback being "The student's answer contains a grammatical error: 'than' should be 'than'" is not smart.

- Special rules (depending on the rubric)
1. For Synthesis and Transformation questions, there are two types of errors: a) Transference errors, and b) Grammar errors.
a) Transference errors:
- Transference errors are errors that does not affect grammar or meaning. 
- Transference errors typically consist of spelling or punctuation error ONLY.
- Examples of transference errors: 1) contractions, 2) spelling error that does not form a different word, and 3) punctuation error (misplacing comma, period, etc). These errors are not considered as changing the grammar/meaning.
- If the rubric penalizes for transference error, then you should follow the marking based on how the rubric would penalize for the occurence of transference errors.
- If the student adds a comma in a place that is not in the ground truth answer, it doesn't change the meaning or grammar, but it is considered as 1 transference error.
- Contractions, if not similar to the provided correct answer/ground truth, are counted as 1 transference error since it does not change the grammar/meaning and should follow the rubric given rubric about the marking guide for the mistake. Example of contractions: using "isn't" when the ground truth uses "is not", using "I'd" when the ground truth uses "I would", "they're", "we'll", etc.
- There is no "transferencee error" that changes the meaning. Any error that changes the meaning is considered as grammar error. So if you want to reason by saying "the student made a transference error that changes the meaning", that is incorrect. You should say "the student made a grammar error that changes the meaning", and should immediately follow the rule of grammar errors stated below.
b) Grammar errors:
- Grammar errors are NOT transference errors. 
- Example of grammar errors: changing verb tense, missing words that changes the meaning, transforming words that changes the meaning, etc. These errors are considered as changing the grammar/meaning, and should be marked according to the rubric (if there are any) about the marking guide for grammar changes in the transformed sentence. 
- Example of grammar errors: the model answer uses "this" but the student answer uses "that", the model answer uses "these" but the student answer changes it into "those", etc. These are considered as grammar errors and not transference errors.
- To check if the mistakes change the grammar & meaning or not, you should properly check with the correct answer/ground truth to see the differences. Differences to be checked are: missing words, transformed words, etc. If there are mistakes in these, then you should follow the given rubric (if there are any) to indicate what to do. 
Here are another several examples of grammar errors (along with the case study example):
-- Punctuation errors like changing/removing apostrophe is counted as grammar error/changing the grammar. An example of punctuation errors that change the grammar is: Instead of "Unless it stops raining, we are not going to John’s house today", the student answers "Unless it stops raining, we are not going to Johns house today", changing "John’s" to Johns"
-- Mistakes in time expressions are considered also as grammatical mistake, which completely changes the meaning, and is considered as missing words that should be transformed from the original sentence. For example, if the transformed sentence required you to use "that night" instead of "tonight", using "tonight" will be counted as that error. Same goes to other time expressions. Another example is using "this" instead of "that" in certain cases. The marking should follow the rule stated by the rubric (if there are any). For example, if the rubric for zero mark states that "The transformed sentence is missing words that should be transformed from the original sentence", that means using "tonight" instead of "that night" is not counted as simple "transference error" anymore since it changes the meaning and the word "that night" is missing from the expected answer.
-- Changing "this" into "that", "these" into "those" or vice versa in for synthesis and transformation or sentence transformation are considered as change of meaning or a grammatical error and is not acceptable. They should NOT be considered as transference error.
--- For example, if the original sentence is "'I want this particular book for my birthday,' Nora told her parents" and the transformed sentence is "Nora told her parents that she wanted this particular book for her birthday", the usage of "this" here should be counted as a mistake since in the reported speech, using "that" will retain the meaning instead of using "this". In this case, compare the exact word used by the correct answer/ground truth. If the usage is different, then you can deem it as incorrect. 
--- Another example is that if the correct answer is "Either the old car or the new motorbikes are parked in the driveway" and the student answers "Either the old car or the new motorbikes are parked in that driveway", we can call it as incorrect since using "that" instead of "the" is incorrect and should be marked according to the rubric for this mistake.
--- Another example is that if the correct answer is "She said that she had lived in that apartment for five years" and the student answers "She said that she had live in this apartment for five years", we can call it as incorrect since using "this" instead of "that" is incorrect and should be marked according to the rubric for this mistake.
--- Another example is that if the correct answer is "My grandmother requested that I help her carry those bags" and the student answers "My grandmother requested that I help her carry these bags", we can call it as incorrect since using "these" instead of "those" is incorrect and should be marked according to the rubric for this mistake.
-- Missing a phrase/word that changes the meaning slightly will still be considered as incorrect (if the rubric suggests so), even though the rest of the sentences are gramamtically correct and structurally sound. For example, if the student answers "Ling wondered why she had not returned the book" instead of "Ling wondered why she had not returned the book on time", the omission of phrase "on time" affects the meaning and should be treated according to the rubric, regardless of whether the rest of the sentences are correct or not.
-- Another examples of grammar errors: changing 'raining' to 'rain', 'houses' to 'house', changing 'John’s' to 'Johns', using 'race car' instead of 'racing car' (which changes the noun phrase)
-- Another examples of errors that forms a different word: changing 'deep' to 'dip', 'very' to 'vary'
-- For the usage of comparative structures, you have to follow the one that is used in the ground truth. For example, if the ground truth uses the "prefer X to Y" structure, that means the student also needs to use "prefer X to Y" structure. If the student uses "prefer X over Y" or "prefer X more than Y" instead, it should be considered as incorrect and be considered as incorrect grammar. This mistake is not considered as transference error and is considered as grammatically incorrect and changes the meaning of the sentence.
-- Omission of comparative structure words is also considered as grammatical error which changes the meaning. For example, if the transformed sentence is "Sharon prefers watching plays to movies" instead of "Sharon prefers watching plays to going to the movies", the omission of "going to the" here is not something that is minor since it changes the meaning. Which means, it is considered as missing a word that should be transformed and should be marked as 0.
-- Difference in past tenses usage is also considered as grammatical error. For example, if the correct answer is "Sarah was offended by his refusal to eat the meal she had prepared" and the student answers "Sarah was offended by his refusal to eat the meal she prepared", then it should be considered as grammar error since the past perfect tense "had prepared" is changed to simple past tense "prepared", which changes the meaning of the sentence. This should be marked according to the rubric for grammar mistakes.
"""
        
        elif any(s in subject for s in ["Physics", "Chemistry", "Biology", "Science"]):
            subject_specific_prompt = """\nThere are also subject specific rules. For this case, you will be handling questions from Science subject.
**Science**
1. Student answers that conveys the same concept as PAS answer and are logically equivalent even though the information inside the answer is explained in opposite terms, you can still consider it as correct.
2. For science questions, any typo, spelling, punctuation, or grammatical errors should not be penalized as long as the meaning stays the same. For example, if the correct answer is "electrolyte" but the student answers "ellectrolyte", it should not be penalized since the meaning is still the same and the typo does not form a different word. Also if the correct answer is "decreases" and the student answers "decrease", it should not be penalized since the meaning is still the same and it does not form a different word.
3. However, if the error changes the meaning, then you should penalize it. For example, if the correct answer is "nucleophilic" but the student answers "nuleofilic" which are two different concepts, then it should be penalized since the meaning is different. Another example is if the correct answer is "constant / none" but the student answers "nothihng". Technically, "nothing" means differently in this case.s
4. Even so, if the phrasing of the fact is different, it should still be considered as correct. For example, if the answer is contextually close to the ground truth or PAS answer, and the student's answer is just slightly unclear, you can still consider it as correct. For example, if the student's answer is "Add number of candle" and the correct answer is "Use more candles instead of one", as they are both contextually close, you should still consider the answer as correct instead of giving half mark.
5. If the correct answer contains multiple alternative answers separated by / sign, and the student is able to mention one of the alternative answers correctly, then they should be marked as full mark, regardless of what the full mark is. For example, if the full mark is 5, they should still get 5 as well. 
6. For science, typos/mispellings in website URLs, sites, etc. in the answer are considered as damaging typos and should be penalized. If the typo is crucial (for example, substance names, species, etc.), then it should be penalized.
7. For science, if the answer is factually correct but is not stated in the model answer/ground truth, you may still consider it as correct. For example, when asked about three materials that are transported around the human transport system and the ground truth is "Oxygen, digested food, waste materials", answering "water" as one of the answers are considered as factually correct which means does not need to be penalized. Which means, even though the answer is not explicitly mentioned in the ground truth, it is a commonly accepted answer and is contextually correct.
"""

        elif any(s in subject for s in ["Maths", "Mathematics", "Math"]):
            subject_specific_prompt = """\nThere are also subject specific rules. For this case, you will be handling questions from Math subject.
**Math**
Rule 1-3 are specified for questions that need the answer to be in the form of number + unit. What counts as a 'unit':
- For length: mm, cm, m, km, etc.
- For mass: mg, g, kg, etc.
- For time: s, min, h, etc.
- For speed: m/s, km/h, etc.
- For money: $, £, etc.
- For other quantities: mol, cd, etc.
- Units can also be in front. For example, if the correct answer is "$10", and the student answers "10", that means their score will be deducted by 0.5 mark, unless the marking note/rubric states otherwise.
What does not count as a 'unit':
- Unofficial metric such as "people", "students", "cars", "trees", "pupils" etc. For example, if the correct answer is "20 students" and the student answers "20", that should not be penalized since "students" is not an official unit. However, if the marking note/rubric states otherwise, you should follow the marking note/rubric.
Rule 1. if the question requires an answer in the form of a number + unit (for example: 20 mm, 13 cm, \\frac{3}{2} km, etc.) and the ground truth also includes number + units, there are several rules you must follow:
Sub-Rule 1: if there are no marking note/rubric that states otherwise, you have to DEDUCT 0.5 mark from the full mark if the student answers only the number (for example: 20, 13, etc.) without the unit. For example, if full mark = 3, then deduct 0.5 so it becomes 2.5. If full mark = 2, deduct 0.5 so it becomes 1.5, and if full mark = 1, deduct 0.5 so it becomes 0.5. This is because the unit is missing. However, if the marking note/rubric states otherwise, you should follow the marking note/rubric. This also applies for units that are in front. For example, if correct answer is $10 and student answers 10 only, that means their score will be deducted by 0.5 mark, unless the marking note/rubric states otherwise.
Sub-Rule 2: if the answer is in different unit representation and is not the same as the unit in the ground truth/correct answer (for example, correct answer is in 2 cm but student answers 20 mm instead of 2 cm, or student answers 1.2 kg instead of 1200 g), you can consider it as correct (unless the marking note/rubric states otherwise), which means you can give full mark. This is because both number and unit are stated, and its just different representation. However, if question specifically mentions about "write your answer in _" which asks for a specific unit, then you should follow it and follow rule 1.
Sub-Rule 3: similar to sub-rule 3, apply the same rule for PAS answer as well. For example, if the PAS answer is "2 cm (1)" or "2 (0.5) cm (0.5)", if the student answers 20 mm instead, you can still consider it as correct. It does not necessarily mean the student has to write it in the same unit representation as the PAS answer unless it is stated so. This also applies to units like speed. For example, if the PAS answer is "1.5 m/s (1)", unless the question explicitly mentions to to present the answer in m/s, if the student answers in km/h or even other units, as long as the conversion is correct, it is still fine.
Sub-Rule 4: different unit spelling representation should not be penalized. For example, writing "millimeter" instead of "millimeters" should not be penalized as it is a minor error.
Rule 2. However, if the ground truth does not contain the unit but the student answers with the unit, you should not penalize the student and give full mark. For example, if the correct answer is "20" and the student answers "20 mm", it should still be considered as correct and get full mark as long as the unit is actually correct, unless the marking note/rubric states otherwise. This is because the student already provides the correct number, and even though they provide an additional information which is the unit, it does not change the correctness of the answer since the number is correct. This also applies for units that are in front. For example, if correct answer is "10" and student answers "$10", that means their score will not be deducted and they will get full mark (provided the unit is also correct), unless the marking note/rubric states otherwise.
Rule 3: if the ground truth contains an alternative answer which looks like this: "Correct answer: 12, alternative answer: 12 m" or "Correct answer: 12 / 12 m", that means if the student does not include the unit, they should still get full mark since the alternative answer already states that the answer can be accepted with or without the unit.
Rule 4. Regarding number representation:
- if the question does not explicitly mention that the answer should be written in a specific way, then different number representation should also be accepted. For example, if the question doesnt necessarily mention the answer has to be in decimal or in a specified significant figure/decimal, then writing "\\(\\sqrt(2)\\) instead of 1.41 should still be considered as correct, as long as the meaning is the same and it is not explicitly mentioned that the answer has to be in decimal. This also applies to other number representations such as fractions, percentages, etc. For example, if the correct answer is "0.5" and the student answers "1/2", it should still be considered as correct since they are the same. 
- however, If the question explicitly mentions that you should write the answer in a specific decimal places/specific significant figures, then you shouldn't accept for answers that don't follow this rule. For example, if the correct answer is 0.67 and the question requires you to write the answer to 2 significant figures or write the answer in 2 decimal places, then you shouldn't accept answers like \\(\\frac{2}{3}\\) instead of 0.67, or even 0.666 instead of 0.67 (since its 3 significant figures/3 decimal places). In this case, if it happens, immediately mark the answer as 0 and explain in your reasoning that the question explicitly requires the answer to be in specific decimal places/significant figures, and the student answer does not follow it.
- Another example is if the question mentions "Write your answer in 2 decimal places" and the correct answer is 0.11, then if the student answers "\\(\\frac{4-\\pi}{8}\\)", it should be considered as incorrect even though the end product of the calculation is 0.11, since the question explicitly requires the answer to be in 2 decimal places and the student answer does not follow it.
- Remember that even if the question does not explicitly mention that the answer should be written in a certain way, if the correct answer contains a unit, then rule 1 is still applied (unless there's a marking note that says otherwise). For example, the question doesn't require the student to write in decimal format, and the correct answer is "0.5 cm", then if the student answers "\\(\\frac{1}{2}\\)", then you should deduct 0.5 based on rule 1 since the student did not write the unit (so only the number is correct).
- Also, if the question does not explicitly require your answer to be in a specific decimal number, if your question is close decimally to the provided correct answer, it should still be considered as correct (the number). For example, if the question does not require you to write in 1 decimal/1 significant figure, and the correct answer is 42.8, if the student answers 42.79, then it should still be considered as correct since its close and it might be because 42.8 is from rounding. However, if the correct answer is "42.8 cm" and the student answers "42.79", then you should still deduct 0.5 according to rule 1 since even thougn we've already established that the number is correct, the answer is still missing the unit (which is required given the correct answer has the unit "cm" attached).
- This also applies if the correct answer is in a much more significant decimal format but the student answer is rounding the decimal. For example, if the question does not require you to write in 3 significant figures, and the correct answer is "125.687", then if the student answers "125.7", which comes from rounding to 1 decimal value, then it should still be considered as correct. However, if the correct answer is "125.687 cm" and the student answers "125.7", then you should still deduct 0.5 according to rule 1 since even thougn we've already established that the number is correct, the answer is still missing the unit (which is required given the correct answer has the unit "cm" attached).
Rule 5. If the ground truth contains space formatting and the student's answer lacks the space formatting, it is still considered as correct as it is just formatting choice. For example: "32 450" is the same as "32450". "\\19\\845" is also the same as "\\19845". This is as long as the digit number is the same.
Rule 6. For answers in fraction, you also have to consider that improper fraction is the same as mixed number. For example, if the correct answer is "6/5" or "\\frac{6}{5}" and the student answers "1 1/5" or "1\\frac{1}{5}", since they are the same, consider them as correct.
Rule 7. If the correct answer is in numerical value but the answer contains a mathematical expression that is equivalent to the numerical value, you can also consider it as correct. For example, if the correct answer is "1.14" but the student answers "\\(\\pi)\\) - 2", since the value of "\\(\\pi)\\) - 2" is approximately 1.14, you can consider it as correct. This is only works if:
- The question does not explicitly mention that the answer has to be in numerical value,
- The usage of \\(\\pi)\\) is acceptable in the context of the question (for example the question requires the student to calculate the area of a circle) and the student is using it in the correct way,
- The mathematical expression provided by the student is actually equivalent to the numerical value in the correct answer
Rule 8. When returning your reasoning, ensure that if the student answers in LaTeX format, you also retain the same format, enclosed in \(..\). For example, instead of "\\frac{6}{5}", you should write it as "\\(\\frac{6}{5}\\)".
"""

        elif any(s in subject for s in ["IGCSE Business"]):
            subject_specific_prompt = """\nThere are also subject specific rules. For this case, you will be handling questions from IGCSE Cambridge Business Studies subject.
Here are subject specific rules for this subject:
1. No 0.5 marks are awarded. So increment is now 1 mark instead of 0.5.
2. If given a marking note, you should prioritize it heavily. For this subject, the marking note will be talking a lot about the marking guideline.
3. You also need to know how to dissect the marking guideline properly. For example, if the marking guideline says "1 mark for defining business term, 1 mark for giving example", then you should know that student needs to give both definition and example to get full mark. If student only gives definition only, they will get 1 mark only. If student only gives example only, they will get 0 mark since they did not define the business term.
4. There are also cases where the student needs to present advantages and disadvantages of something.
"""
        
        else:
            subject_specific_prompt = ""

        return subject_specific_prompt

    async def _construct_rubric_prompt(self):
        """
        Rubric Prompt
        """
        ## Get Rubric Prompt
        if self.rubric:
            rubric_prompt = "\nAlong with the PAS answer, ground truth answer and all the stuffs that helps you in making the marking process better, you will also be supported with a general rubric that will set up a general rule for a certain subject (a subject is, for example, English, Chinese, Science, Math, etc.). The user will give you the rubric to follow. However, should there be a contradiction between the rubric and the PAS answer labelling (for example, in the rubric it says not to penalize grammar mistakes but the PAS answer penalizes it), you should prioritize PAS answer first. The rubric will have three scoring criteria: full mark, half mark, and zero mark.\n"
        else:
            rubric_prompt = ""

        return rubric_prompt

    async def _construct_question_type_prompt(self):
        """
        Special prompts depending on the question type (FITB or OEQ)
        Also includes subject specific prompt
        """

        ## Get Subject Specific Prompt and Rubric Prompt
        subject_specific_prompt = await self._construct_subject_specific_prompt()
        rubric_prompt = await self._construct_rubric_prompt()


        if self.type_fitb: ## For Fill in the Blank questions
            question_type_prompt = """{subject_specific_prompt}
For this type of question, you will be focusing on Fill in the Blank questions. The question will have a blank part flagged by (blank _). 
Both the students answers as well as the correct answers will be in form of 'blank _ : answer'.
{rubric_prompt}

For this problem you will be given, only give mark for blank index number {index}. For example, "Blank 1" means it has 1 as the index number.
Which means, only return your answer for this index only, even if there are more than one indexes. The answer can still have alternative answers, where the format will be: "Blank 1: (answer 1) / Blank 1: (alternative answer)". If there are 6 alternative answers, each of them should be considered as correct for this case. Even if you think the answer is not applicable for that blank, if it is listed as an alternative answer, you should still consider it as correct.
If the ground truth comes in more than one filled blanks answers, only use the correct index as your baseline for the answer to be compared. Student answer for Blank 1 should be compared with ground truth for blank 1 answer only.
This means, according to the defined maximum score, you should also give the answer the maximum score if the answer is correct compared to the ground truth.

When the question asks you to fill in with the correct spelling (or ejaan in Malay), you need to look into the correct answer and DO NOT manipulate the spelling in both the correct answer and the student's answer. For example, if the incorrect word is "Jeket" and the correct answer for that blank is "Jaket", when student answers "Jaket", do not manipulate the student's answer into "Jeket" or any other thing, indicating that the student's answer is correct.

If the question type is Synthesis and Transformation or Sentence Transformation, you also have to look into the location of the blank. If the blank to be filled is in the middle or at the end of the sentence, then the rule of capitalization is also applied to the first letter of the student's answer. For example, if the sentence/question to be filled is: "Everyone (blank 1)" and the student's answer is "Adores the baby since she is cute", that means that since the filled sentence is "Everyone Adores the baby since she is cute", which is not correct in terms of capitalization since it should be "adores" instead of "Adores" in this case. However, if the blank is located at the start of the sentence, then it has to began with upper case letter. For example: "(Blank 1) is the most important thing to see", which indicates that the blank that needs to be filled is at the beginning of the sentence.
    """.format(
        subject_specific_prompt = subject_specific_prompt, 
        rubric_prompt = rubric_prompt, 
        index = self.fitb_index
    )

        else: ## Preferably for Open Ended Questions
            question_type_prompt = """{subject_specific_prompt}
{rubric_prompt}
""".format(
    subject_specific_prompt = subject_specific_prompt, 
    rubric_prompt = rubric_prompt
)

        return question_type_prompt

    async def _construct_response_prompt(self):
        ## Get Response Prompt
        response_prompt = "\n\nWhen giving your response, you should also state the reason for giving the mark. When doing this, you should align with your given mark and do not contradict."
        response_prompt += " The reason might come from the rubric, or from your own judgement (preferably from the rubric)" if self.rubric else ""

        return response_prompt

    async def _construct_system_prompt(self):
        general_rules_prompt = await self._construct_general_rules_prompt()
        special_parameter_prompt = await self._construct_special_parameter_prompt()
        question_type_prompt = await self._construct_question_type_prompt()
        response_prompt = await self._construct_response_prompt()

        system_prompt = """You are an assistant that is responsible on acting as an automarker system.
             
You will be given two answers; one which acts as the ground truth, and the other one which comes from a student. The purpose of this automarking system is to mark the student's answer by comparing it to the given ground truth.

There are two types of ground truths:
1. One that is the original correct answer. We call this as "Ground Truth Answer"
2. One that has been annotated with a mark given by another AI. We call this as "Point Allocated System (PAS) answer"

PAS answer is optional, and the user will tell you whether "PAS answer" exists or not. Ground Truth Answer, however, is absolute and will always exist.

Read and apply all specific instructions below; always reason step-by-step before awarding marks; if information from marking notes, incorrect flags, or rubrics conflicts with general rules, follow this strict order of precedence:
**Marking Note** > **Rubric** > **General Rules**

## Marking Inputs and Parameters
You will be provided with:
- The question text, included with indicator such as (blank 1) if the question is Fill in the Blank
- Ground Truth Answer (always present)
- PAS answer (optional): AI-annotated, with possible partial marks
- Incorrect Flags (optional): answers/terms to mark as strictly incorrect
- Marking Note (optional): question-specific marking exceptions or instructions
- Rubric (optional): general additional marking guidelines
- Student answer to grade
- Maximum score (provided by user)

**Score increment:** 0.5 if partial scoring is enabled (general default). Lower bound is always 0; upper bound is maximum score provided.

## General Marking Logic
1. There are two types of scores that can be given to the answer:
- Partial scores
- Full scores
2. **Partial scoring**: The default mode is "Partial scores", meaning we can give half mark to the answer (for example, a full mark of 1 will have 0.5 as its half mark). The PAS answer for this mode can be identified by having annotated marks that are less than the full mark. For example: "answer 1 (0.5), answer 2 (0.5)" when the full mark is 1.
3. **Full scores mode**: However, sometimes the Ground Truth Answer are separated by the / symbol. This symbol will indicate that the answer has more than one possibility, if this happens, switch to "Full scores" mode and each alternative answer will be considered as full mark, regardless of what the subject of the question is (English, Chinese, Math, Science, etc.). Meaning that if the correct answer looks like this: "answer 1 / answer 2 / answer 3", it indicates that there are three possible correct answers that you can use as the reference. If the format is "answer 1 /\nanswer 2 /\nanswer 3 /\nanswer 4", each of them are alternative answers. However, even if student's answer is not in any of the alternative answers, it is immediately incorrect (unless the meaning is also similar to one of the answer possibilities). This is because there are so many answer possibilities especially for English subject questions.
{general_rules_prompt} {special_parameter_prompt}

You will give mark to the student answer with the minimum score of 0. User will further tell you what the maximum score is. The increment will be {step} if using "partial mode". This means that you cannot give mark that is out of the boundary.
{question_type_prompt} {response_prompt}

Return your response in the following JSON format:
{{
    "student_answer": the student's answer. Ensure any LaTeX format that is given by the input is not changed and kept.,
    "mark": score of the answer, following the minimum and maximum mark as well as the increment.
    "reason": the reason behind the mark allocation for the student's answer, based on the given correct answer.
}}

For the "student_answer" field, make sure that if the given student's answer is in LaTeX format. For example, if the given student's answer is wrapped in \\(..\\), then do not change it to \(..\)
""".format(
    general_rules_prompt = general_rules_prompt, 
    special_parameter_prompt = special_parameter_prompt, 
    step = self.step, 
    question_type_prompt = question_type_prompt, 
    response_prompt = response_prompt
)

        return system_prompt

class AutomarkingUserPromptConstructor:
    """
    Class to construct user prompt for automarking
    """
    def __init__(self, messages: List[dict], question_context: QuestionContext, is_multi: bool = False):
        self.messages = messages
        self.question_context = question_context
        self.vectorstore = False
        self.is_multi = is_multi

        ## define the rest of the parameters
        self.question = question_context.question
        self.string_urls = question_context.string_urls
        self.subject = question_context.subject
        self.language = question_context.language
        self.incorrect_flags = question_context.incorrect_flags
        self.marking_note = question_context.marking_note
        
        self.correct_answer = question_context.answer_context.correct_answer
        self.pas_answer = question_context.answer_context.pas_answer
        self.student_answer = question_context.answer_context.student_answer
        self.student_image_urls = question_context.answer_context.student_image_urls
        self.solution = question_context.answer_context.solution
        
        self.type_fitb = question_context.mark_context.type_fitb
        self.full_mark = question_context.mark_context.full_mark
        self.fitb_index = question_context.mark_context.fitb_index
        self.num_blanks = question_context.mark_context.num_blanks
        self.unique_blanks = question_context.mark_context.unique_blanks

        self.rubric = question_context.rubric_context.string_rubric ## use the cleaned string version of the rubric, instead of the JSON object version

    def _is_image_url(self, url):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg')

        if isinstance(url, list):
            url = url[0] if url else ""
        return url.lower().startswith(('http://', 'https://')) and url.lower().endswith(image_extensions)

    async def _append_prompt(self, messages, type: Literal['text', 'image_url']):
        self.messages.append(await get_user_prompt(
            message = messages,
            type = type,
            vectorstore = self.vectorstore
        ))

    async def _append_question_prompt(self):
        ## Question
        if not self.is_multi:
            await self._append_prompt(f"Here is the composition & question:\n{self.question}", "text")
        else:
            await self._append_prompt(f"Here is the composition & question:\n{self.question}\n================\nClearly look at the instruction inside the question. If the question asks you to answer in only one word, then answers that are not one word will be considered incorrect.", "text")

        ## Question Images (Optional)
        for item in self.question_context.string_urls:
            await self._append_prompt(item, "image_url")

    async def _append_model_answer_prompt(self):
        question_extractor = QuestionExtractor()
        is_image_answer = False
        correct_answer_image = []
        additional_model_answer_prompt = ""

        if not self.correct_answer and self.solution:
            self.correct_answer = self.solution
            print("Using solution as correct answer since correct answer is not provided.")

        ## Model Answer
        if not self.is_multi:
            correct_answer = ""
            if isinstance(self.correct_answer, list):
                # correct_answer = " / ".join(self.correct_answer)
                try:
                    correct_answer += f"Correct answer: {self.correct_answer[0]}"
                except IndexError:
                    correct_answer += "Correct answer: "
                if len(self.correct_answer) > 1:
                    for index, item in enumerate(self.correct_answer[1:], start=1):
                        correct_answer += f"\nAlternative correct answer {index}: {item}"
            else:
                correct_answer = self.correct_answer

            if self.num_blanks != 0: ## if FITB
                model_answer_prompt = f"Here are the correct answer and alternative answers that are accepted:\n{correct_answer} | There are {self.num_blanks} acceptable correct answers. If it matches one of the correct answers, we can consider it as correct. The full mark is {self.full_mark}."
                
                if self.unique_blanks <= 2: ## if FITB and more than one blank
                    answer_prompt = f"Here are the correct answer and alternative answers that are accepted:\n{correct_answer}"
                    answer_list = await question_extractor.extract_fitb_answers(text = correct_answer)
                    for index, item in enumerate(answer_list):
                        answer_prompt += f"\n\nCorrect answer used in full sentence {index + 1}: {await question_extractor.replace_blank(self.question, item, self.fitb_index)}"
                    model_answer_prompt = f"{answer_prompt}\n\nThere are {self.num_blanks} acceptable correct answers. If it matches one of the correct answers, we can consider it as correct. The full mark is {self.full_mark}."
                
            else: ## if not FITB
                ## if subject is IGCSE Business, add more explanation
                if "IGCSE Business" in self.subject:
                    model_answer_prompt = f"Here is the ground truth/model answer: {correct_answer} | In this subject, there are no 0.5 marks awarded, so the student will either get full mark or zero mark only. The full mark is {self.full_mark}.\n\nAlso, if the correct answer contains [k], [app] or [an], remember that each point made in the student's answer cannot be reused for another point."
                else:
                    model_answer_prompt = f"Here is the ground truth/model answer: {correct_answer} | The full mark is {self.full_mark}. | Do NOT mistaken correct answer as the student's answer. You will use this correct answer to be compared with the student answer."
                
                for index, item in enumerate(self.correct_answer) if isinstance(self.correct_answer, list) else enumerate([self.correct_answer]):
                    if self._is_image_url(item):
                        correct_answer_image.append(item)
                        additional_model_answer_prompt = f"\nThe correct answer is in form of image. Look at the image carefully to understand the correct answer and compare it with the student answer."
                        is_image_answer = True
                    if is_image_answer:
                        model_answer_prompt += f"{additional_model_answer_prompt}"

                if "accept any" in correct_answer.lower() or "accept all" in correct_answer.lower():
                    model_answer_prompt += f"\n\nif the correct answer specifically mentions to accept any N of a set of answers, then if the student answers N of them correctly, award {self.full_mark} instead of half mark. For example, if the correct answer mentions to accept any 2 of a set of answers correctly, then if the student answers 2 of them correctly, award {self.full_mark} instead of half mark."
                    
        else:
            model_answer_prompt = "Here are all the correct answers for all the blanks:"
            for index, answer in enumerate(self.correct_answer):
                model_answer_prompt += f"\nCorrect Answer for {answer}"
            model_answer_prompt += f"\n\nThe full mark for the whole question is: {self.full_mark}."

        ## PAS Answer
        if self.pas_answer:
            if isinstance(self.pas_answer, list):
                pas_answer_prompt = f"PAS answer exists. You still need to refer to the ground truth/model answer in case there are alternative answers that are not put in the PAS answer. Here are the PAS answer:"
                for index, item in enumerate(self.pas_answer):
                    pas_answer_prompt += f"\n\nPAS Answer {index + 1}: {item}"
            else:
                pas_answer_prompt = f"PAS answer exists. You still need to refer to the ground truth/model answer in case there are alternative answers that are not put in the PAS answer. Here is the PAS answer: {self.pas_answer}\nRemember, if the PAS answer is in form of 'answer 1 / answer 2 / answer 3 (1) answer 1 / answer 2 / answer 3 (1) while the full mark is 2 instead of 1, that means you are expected to just mention 2 out of 3 items at minimum to get full mark (2). Which means, if the student answers 'answer 1 and answer 2', that is considered as correct. If the student is able to answer all three, we don't penalize either."
        else:
            pas_answer_prompt = f"PAS answer does not exist. Refer to the ground truth/model answer only."
       
        await self._append_prompt(model_answer_prompt, "text")
        if is_image_answer:
            for image_url in correct_answer_image:
                await self._append_prompt(image_url, "image_url")
        await self._append_prompt(pas_answer_prompt, "text")
        

    async def _append_rubric_prompt(self):
        ## Rubric
        if self.rubric:
            rubric_prompt = f"Here is the general rubric to also help you in giving the marks. You can prioritize rules inside the rubrics instead of general rules or subject rules:\n{str(self.rubric)}"
            if "0.5 marks are not allowed" in self.rubric:
                possible_mark_range = ', '.join(str(i) for i in range(int(self.full_mark) + 1))
                rubric_prompt += f"0.5 marks are not allowed, so the mark possibility is only {possible_mark_range}."
        else:
            rubric_prompt = "No general rubric for this case. You can either follow the general rules or the marking note (if there are any)."

        await self._append_prompt(rubric_prompt, "text")

    async def _append_additional_rules_prompt(self):
        """for both incorrect flags and marking note"""
        
        if self.incorrect_flags:
            incorrect_flags_prompt = f"Incorrect flags exist. Here are the incorrect flags:"
            for incorrect_flags_index, item in enumerate(self.incorrect_flags):
                incorrect_flags_prompt += f"\nIncorrect flag {incorrect_flags_index + 1}: {item}"
            incorrect_flags_prompt += f"\nStudent will get no additional mark for mentioning this part."
        else:
            incorrect_flags_prompt = "Incorrect flags does not exist. No need to worry about this part."
        
        if self.marking_note:
            marking_note_prompt = f"Marking note exists. Here are the marking notes: {self.marking_note}."
            
            ## additional rule: if marking note is talking about "typo", then an additional system must be added to calculate normally
            if "typo" in self.marking_note.lower():
                correct_answer = ""
                if isinstance(self.correct_answer, list):
                    correct_answer = self.correct_answer[0] ## just take the first correct answer if there are more than one correct answers, since we just want to use it as a reference for typo analysis
                else:
                    correct_answer = self.correct_answer

                question_extractor = QuestionExtractor()

                # Clean to remove \(..\) answers
                self.student_answer = await question_extractor.clean_student_answer(self.student_answer)

                if self.type_fitb:
                    model_answer_list = await question_extractor.extract_model_answers_fitb(
                        correct_answer = correct_answer,
                        fitb_index = self.fitb_index
                    )
                else:
                    model_answer_list = await question_extractor.extract_model_answers(
                        correct_answer = correct_answer
                    )
                
                error_list = []
                for model_answer in model_answer_list:
                    typo_analysis = await question_extractor.analyze_typos(
                        correct_answer = model_answer,
                        student_answer = self.student_answer
                    )
                    typo_analysis['model_answer'] = model_answer
                    error_list.append(typo_analysis)
            
                if error_list:
                    marking_note_prompt += f"\n\nAdditionally, since the marking note mentions 'typo', please consider the following typo analysis when giving marks:\n\nStudent's answer: {self.student_answer}"
                    for error_index, item in enumerate(error_list):
                        marking_note_prompt += f"\n\nTypo Analysis {error_index + 1}:\nUsing Model Answer {error_index + 1}: {item['model_answer']}\nTotal typos found: {item['total_errors']}\nInsertions: {item['breakdown']['insertions']}\nSubstitutions: {item['breakdown']['substitutions']}\nDeletions: {item['breakdown']['deletions']}"

                    marking_note_prompt += f"\n\nUse the lowest typo count from the above analyses when considering marks deduction for typos. For example, if compared with Model Answer 1 there are 2 typos, and compared with Model Answer 2 there are 3 typos, then use 2 typos to indicate whether marks should be deducted based on the marking note."
                    marking_note_prompt += f"\n\nAs an additional rule, if the marking note specifies to accept only answer with 1 word, then you still need to look at the correct answer/PAS answer provided. If the correct answer/PAS answer has two words, but the marking note says only accept answer with 1 word, then if the student answer has 2 words, it should still be considered as correct since the correct answer already has 2 words. However, if the correct answer/PAS answer has only 1 word, and the marking note says only accept answer with 1 word, then if the student answer has 2 words, it should be considered as incorrect since the correct answer only has 1 word and the marking note says only accept answer with 1 word."
        else:
            marking_note_prompt = "Marking note does not exist. Use the general rules for the marking instead."
        
        await self._append_prompt(incorrect_flags_prompt, "text")
        await self._append_prompt(marking_note_prompt, "text")

    async def _append_student_answer_prompt(self):
        question_extractor = QuestionExtractor()

        if self.student_image_urls:
            # Student Answers are in form of images
            if isinstance(self.student_image_urls, str):
                self.student_image_urls = [self.student_image_urls]
            
            student_answer_prompt = """## Image-Based Answer Evaluation
The student has submitted an image as their answer. When evaluating image-based answers:
1. Carefully analyze the visual content of the image (diagrams, graphs, handwritten work, drawings, etc.)
2. For diagrams/graphs: evaluate accuracy of axes labels, curves, shifts, equilibrium points, arrows, annotations, etc.
3. For handwritten answers: read and interpret the written content accurately
4. Compare the image content against the model answer requirements
5. Award marks based on how well the image demonstrates understanding of the concepts
6. In your reason, describe what you observed in the image and how it relates to the marking criteria

The student has submitted the following image(s) as their answer. Please evaluate the content of the image(s):"""
            await self._append_prompt(student_answer_prompt, "text")

            for img_url in self.student_image_urls:
                # Append each image URL as an image_url type prompt
                await self._append_prompt(img_url, "image_url")

        else:
            # Student Answers are in form of texts
            if not self.is_multi:
                if self.type_fitb: ## if FITB
                    student_answer_prompt = f"Here is the student's answer:\nblank {self.fitb_index}: {self.student_answer} | Do not mistakenly alter the student's answer (whether in terms of spelling or others)"
                    if self.unique_blanks <= 2: ## if FITB and more than one blank
                        student_answer_prompt = f"Here is the student's answer:\nblank {self.fitb_index}: {self.student_answer} | Full sentence: {await question_extractor.replace_blank(self.question, self.student_answer, self.fitb_index)}\n\nDo not mistakenly alter the student's answer (whether in terms of spelling or others)\n\nIf the question requires the student's answer to provide a question mark, the student's answer does not need to provide a question mark if the question statement already includes one after the blank. Can also see in the above's full sentence provided."
                
                else: ## if not FITB
                    student_answer_prompt = f"The student's answer for this question is: {self.student_answer} | Do not mistakenly alter the student's answer (whether in terms of spelling or others). When returning the student's answer in the JSON output for the \"student_answer\" field, ensure that any LaTeX format is kept the same."
                    # The student answer is transcribed from a photo of the whole
                    # page, so it can include the question/instructions printed on
                    # it. Those are not the student's response -- mark only the
                    # answer. The actual question is supplied separately above.
                    if self.question:
                        student_answer_prompt += " | IMPORTANT: The text above was transcribed from a photo of the page, so it may include the question, instructions, title, or numbering that was already on the page. That is NOT the student's response. Do not mark, reward, or penalise that text; evaluate ONLY the student's own answer to the question. The question you are marking against is the one given to you separately above."
                
                if any(s in self.subject for s in ["Math", "Maths", "Mathematics"]):
                    if any(s in self.correct_answer for s in ["cm"]):
                        student_answer_prompt += f"\n\nPlease also check if the question requires the student to write an answer with a measurement unit or not. If the correct answer has an unit attached to it (e.g. '20 cm') but the student's answer doesn't, then follow the rule provided in the system prompt."

                await self._append_prompt(student_answer_prompt, "text")
                
                if self.fitb_index != 0:
                    await self._append_prompt(f"Give mark to answer blank {self.fitb_index} only", "text")
            else:
                student_answer_prompt = f"Here are the student's answer for all the blanks:"
                
                for index, answer in enumerate(self.student_answer):
                    student_answer_prompt += f"\nStudent Answer for blank {index + 1}: {answer}"
                
                student_answer_prompt += f" | Full Sentence: {await question_extractor.replace_blank(self.question, self.student_answer, self.fitb_index)}"
                student_answer_prompt += f"\n\nDo not mistakenly alter the student's answer (whether in terms of spelling or others)"

                if any(s in self.subject for s in ["Math", "Maths", "Mathematics"]):
                    if any(s in self.correct_answer for s in ["cm"]):
                        student_answer_prompt += f"\n\nPlease also check if the question requires the student to write an answer with a measurement unit or not. If the correct answer has an unit attached to it (e.g. '20 cm') but the student's answer doesn't, then follow the rule provided in the system prompt."
                
                await self._append_prompt(student_answer_prompt, "text")

            # if any(s in self.subject for s in ["Math", "Maths", "Mathematics"]):
            #     if any(s in self.correct_answer for s in [" cm"]):
            #         if " cm" not in self.student_answer:
            #             await self._append_prompt("The correct answer contains the unit 'cm', but the student's answer does not contain 'cm'. This means that even if the number provided by the student is correct, you should still deduct 0.5 mark since the student did not provide the unit, which is required given the correct answer has the unit 'cm' attached.", "text")

        if self.student_answer == "":
            await self._append_prompt("The student might have not provided any answer in this case.", "text")

    async def _append_ai_marking_result_prompt(self, marking_results: List[dict] = []):
        ai_marking_prompt = f"Here are the previous AI marking results (with 'Mark per Blank' scheme):\n"
        for index, item in enumerate(marking_results):
            ai_marking_prompt += f"\n{index + 1}. Blank {index + 1}\nAnswer: {item['student_answer']}\nMark: {item['mark']}\nReason: {item['reason']}"

        await self._append_prompt(ai_marking_prompt, "text")

    async def _append_language_prompt(self):
        language = "english" if not self.language else self.language.lower()
        if language.lower() != "english":
            await self._append_prompt(f"When giving your reasoning, use {language} instead of english. However, refer to parts of the answers in the original language. For example, if the question is in English and you are asked to output your reasoning in Vietnamese, you should still use Vietnamese, but refer to parts of the student answer using the original version (english), instead of translating the answer to Vietnamese", "text")

    async def _construct_messages_object(self, marking_results: List[dict] = []):
        await self._append_question_prompt()
        await self._append_model_answer_prompt()
        await self._append_rubric_prompt()
        await self._append_additional_rules_prompt()
        await self._append_student_answer_prompt()
        if self.is_multi:
            await self._append_ai_marking_result_prompt(marking_results)
        await self._append_language_prompt()

        return self.messages

class MarkingResultContext:
    def __init__(
        self,
        ai_mark: Union[float, int] = 0,
        reason: str = ""
    ) -> None:
        self.ai_mark = ai_mark
        self.reason = reason

class RubricTransformer:
    def __init__(self, reqs: AutoMarking = None):
        self.reqs = reqs
        pass

    def _is_multiple(self, value: Decimal, step: Decimal) -> bool:
        # True if value is an exact multiple of step
        q = value / step
        return q == q.to_integral_value()

    def _dec_to_str(self, d: Decimal) -> str:
        # Nice formatting: 2.0 -> "2", 0.5 -> "0.5"
        return str(int(d)) if d == d.to_integral() else format(d.normalize(), 'f')

    def convert_mark_string(self, mark_str: str, full_score=None, step=0.5) -> str:
        # Replace underscores with spaces
        title = mark_str.replace('_', ' ')
        suffix = ""

        if full_score is not None and mark_str in ("full_mark", "half_mark", "zero_mark"):
            d_full = Decimal(str(full_score))
            d_step = Decimal(str(step))
            d_half = d_full / Decimal(2)

            # Only annotate if half is representable with given step and full >= step
            if d_full >= d_step and self._is_multiple(d_half, d_step):
                if mark_str == "full_mark":
                    v = d_full
                elif mark_str == "half_mark":
                    v = d_half
                else:  # "zero_mark"
                    v = Decimal(0)
                suffix = f" ({self._dec_to_str(v)} mark)"

        # Add a colon and space at the end
        return f"# Give {title}{suffix} if:"

    def render_rubric(self, rubric, full_score=None, step=0.5) -> str:
        string_rubric = ""
        for key in rubric[0]:
            if key == "note":
                for item in rubric[0][key]:
                    string_rubric += f"**{str(key)}:** {item}\n"
            else:
                key_string = self.convert_mark_string(str(key), full_score=full_score, step=step)
                string_rubric += key_string
                string_rubric += f"\n{rubric[0][key]}\n"
        return string_rubric
    
    def constructRubricContext(self) -> RubricContext:
        reqs = self.reqs

        rubric_json = []
        string_rubric = ""

        if reqs.rubric != "":
            rubric_json = json.loads(reqs.rubric)
            string_rubric = self.render_rubric(rubric_json, full_score = reqs.full_mark, step = 0.5)

        return RubricContext(
            rubric = reqs.rubric,
            rubric_json = rubric_json,
            string_rubric = string_rubric
        )

class QuestionExtractor:
    def __init__(self, reqs: AutoMarking = None):
        self.reqs = reqs

    async def _replace_ans_tags(self, html_string):
        # Parse the HTML string
        soup = BeautifulSoup(html_string, 'html.parser')

        ## Find all <span ans> tags
        ans_spans = soup.find_all('span', class_="ans-div")

        for idx, span in enumerate(ans_spans, start=1):
            # replace entire span with the blank text, discarding content
            span.replace_with(f"(blank {idx})")
        
        return str(soup)

    async def _correct_url(self, url):
        parts = url.split('/')
        clean_parts = []
        for part in parts:
            if part not in clean_parts:
                clean_parts.append(part)
        return '/'.join(clean_parts)

    async def _filter_and_correct_urls(self, urls):
        url_pattern = re.compile(r'https://static-contents-smartjen\.s3\.ap-southeast-1\.amazonaws\.com/img/(answerImage|articleImage|questionImage|workingImage)/[a-zA-Z0-9_-]+\.[a-zA-Z0-9]{3,4}')

        corrected_urls = []
        for url in urls:
            corrected_url = await self._correct_url(url)
            if url_pattern.match(corrected_url):
                corrected_urls.append(corrected_url)
        return list(set(corrected_urls))  # Removing duplicates

    async def _convert_mark_string(self, mark_str):
        # Replace underscores with spaces
        mark_str = mark_str.replace('_', ' ')
        # Add a colon and space at the end
        return mark_str + ": "

    async def _convert_fitb_answer(self, correct_answer):
        # 1) Find all segments of the form "blank # : <text>", non-greedily, until the
        #    next "blank # : " or end-of-string. This avoids cutting off at 'b'.
        segments = re.findall(
            r'(blank\s+\d+\s*:\s+.*?)(?=blank\s+\d+\s*:|$)',
            correct_answer
        )

        # 2) Collect all answers associated with each "blank N".
        #    E.g. answers["blank 1"] = ["Water", "Carbon dioxide"]
        answers = {}
        for seg in segments:
            match = re.match(r'(blank\s+\d+)\s*:\s*(.*)', seg.strip())
            if match:
                key = match.group(1)     # e.g., "blank 1"
                value = match.group(2)   # e.g., "Water"
                answers.setdefault(key, []).append(value)

        # 3) Build the final formatted string.
        #    For each blank N, join repeated answers with " / blank N : " 
        #    and place each blank's answers on its own line.
        result_lines = []
        # Sort keys by the number after "blank "
        def blank_number(blank_key):
            # Extract the integer after 'blank'
            return int(re.search(r'(\d+)', blank_key).group(1))

        for key in sorted(answers.keys(), key=blank_number):
            # Example: if answers[key] is ["Water", "Carbon dioxide"] 
            # we want: "blank 1 : Water / blank 1 : Carbon dioxide"
            combined = " / ".join(f"{key} : {val}" for val in answers[key])
            result_lines.append(combined)

        return "\n".join(result_lines)

    async def _transform_blank_answers(self, text, blank_n):
        # Split the text by the pattern
        parts = text.split(f"blank {blank_n} : ")
        
        # Remove empty strings and clean up slashes and whitespace
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            if part:
                # Remove trailing slash if it exists
                if part.endswith("/"):
                    part = part[:-1].strip()
                cleaned_parts.append(part)

        if not cleaned_parts:
            return text, 0

        num_blanks = len(cleaned_parts)
        
        # Construct the formatted result
        output = [f"correct answer: {cleaned_parts[0]}"]
        output = []
        for i, part in enumerate(cleaned_parts[0:], 1):
            output.append(f"correct answer {i}: {part}")
        
        return "\n".join(output), num_blanks

    async def _extract_blank_section(self, text: str, blank_n: int):
        ## get all the unique blank numbers in the text (to count the total blanks)
        blank_numbers = re.findall(r'blank\s*(\d+)\s*:', text, re.IGNORECASE)
        unique_blanks = len(set(blank_numbers))

        ## Get the the corresponding blank section and the alternatives from the given text
        pattern = rf"blank {blank_n} :.*?(?=\nblank {blank_n + 1} :|$)"
        match = re.search(pattern, text, re.DOTALL)

        full_text = f"{match.group(0)}" if match else text
        text_result, num_blanks = await self._transform_blank_answers(full_text, blank_n)

        return full_text, num_blanks, unique_blanks

    async def extract_blank_section_dispatch(self, text: str, blank_n: Union[int, List[int]]):
        """
        If blank_n is a list, returns a list of _extract_blank_section result tuples (full_text, num_blanks, unique_blanks).
        Otherwise, returns a single result.
        """
        if isinstance(blank_n, list):
            print("Multiple blank_n detected.")
            # Multiple blank_n, run for each
            extracted_answers = []
            extracted_num_blanks = []
            
            for b in blank_n:
                res = await self._extract_blank_section(text, b)
                extracted_answers.append(res[0])
                extracted_num_blanks.append(res[1])
            return extracted_answers, extracted_num_blanks, res[2]
        else:
            res = await self._extract_blank_section(text, blank_n)
            return res[0], res[1], res[2]

    async def extract_fitb_answers(self, text: str):
        # Split by '/'
        items = text.split('/')
        result = []
        for item in items:
            # Remove leading/trailing whitespace
            item = item.strip()
            # Remove 'blank [number] :' prefix using regex
            sentence = re.sub(r'^blank \d+ *: *', '', item, flags=re.IGNORECASE)
            result.append(sentence)
        return result

    async def replace_blank(self, text: str, replacement: Union[str,List[str]], blank_index: Union[int,List[int]]):
        """
        Replaces (blank n) with replacement string(s) in the given text.

        :param text: The original string with placeholders like (blank 1), (blank 2), etc.
        :param replacement: The string(s) to replace the placeholder(s) with.
        :param blank_index: The index number(s) that indicate which (blank n) to replace.
        :return: The string with the indicated blank(s) replaced.
        """

        try:
            if isinstance(blank_index, list) and isinstance(replacement, list):
                result = text
                for idx, repl in zip(blank_index, replacement):
                    pattern = r'\(blank {}\)'.format(re.escape(str(idx)))
                    result = re.sub(pattern, repl, result, count=1)
                return result
            else:
                # Single replacement -- original behavior
                pattern = r'\(blank {}\)'.format(re.escape(str(blank_index)))
                result = re.sub(pattern, replacement, text, count=1)
                return result
        except Exception as e:
            return text

    async def _get_answers(self, question):
        answer_options = ""
        correct_answer = ""
        mark = ""
        answer_option_pattern = re.compile(r"answer option:(.*?)correct answer:", re.DOTALL)
        correct_answer_pattern = re.compile(r"correct answer:(.*?)mark:", re.DOTALL)
        mark_pattern = re.compile(r"mark:(.*)Solution:", re.DOTALL)
        solution_pattern = re.compile(r"Solution:(.*)", re.DOTALL)

        answer_option_match = answer_option_pattern.search(question)
        correct_answer_match = correct_answer_pattern.search(question)
        mark_match = mark_pattern.search(question)
        solution_match = solution_pattern.search(question)

        if answer_option_match:
            answer_options = answer_option_match.group(1).strip()

        if correct_answer_match:
            correct_answer = correct_answer_match.group(1).strip()

        if mark_match:
            mark = mark_match.group(1).strip()

        if solution_match:
            solution = solution_match.group(1).strip()

        return answer_options, correct_answer, mark, solution

    async def extract_question_obj(self, data):
        ## Predefine variables
        string_question = ""
        string_urls = []
        string_urls_solutions = []
        answer_options = []
        answer_options_list = []
        correct_answer = None
        solution = None
        img_url_pattern = re.compile(r'src=[\'"]?([^\'" >]+)')
        skip_conditions = ["question type", "question header", "difficulty_level"]
        break_conditions = ["Solution"]
        solution_images = []

        is_answer_option = False
        is_solution = False

        type_fitb = False
        type_mcq = False

        question_type = data[0]['text']
        if any(word.lower() in question_type.lower() for word in ["FITB Without Option", "Drag & Drop"]):
            type_fitb = True
            type_mcq = False
        elif "MCQ" in question_type:
            type_mcq = True
            type_fitb = False
        else:
            type_fitb = False
            type_mcq = False

        for item in data:
            if item['type'] == 'image_url':
                image_url = item['image_url']
                if isinstance(image_url, dict):
                    if not is_solution:
                        string_urls.append(image_url['url'])
                    else:
                        string_urls_solutions.append(image_url['url'])
                elif isinstance(image_url, list):
                    for url_item in image_url:
                        if not is_solution:
                            string_urls.append(url_item['url'])
                        else:
                            string_urls_solutions.append(url_item['url'])
            elif item['type'] == 'text':
                text = item['text']
                ## if question is of type mcq, there might be images in the answer options/correct answer as well
                if not type_mcq and "img style" in text and not is_solution and not is_answer_option:
                    match = img_url_pattern.search(text)
                    if match:
                        string_urls.append(match.group(1))
                else:
                    if any(cond in text for cond in skip_conditions):
                        continue
                    else:
                        if "FITB Without Option" in data[0]['text']:
                            text = await self._replace_ans_tags(text)

                        if "correct answer" in text:
                            is_answer_option = False

                        if is_answer_option:
                            text = text + " $$"

                        if "answer option" in text:
                            is_answer_option = True

                        if "Solution:" in text:
                            is_solution = True
                        
                        string_question += text + " "

        ## Get Correct Answer and Answer Options
        answer_options, correct_answer, mark, solution = await self._get_answers(string_question)

        ## clean up answer options, correct answer and solution
        answer_options  = answer_options.replace('<br>', '\n').replace('<br />', '\n')
        correct_answer  = correct_answer.replace('<br>', '\n').replace('<br />', '\n')
        solution = solution.replace('<br>', '\n').replace('<br />', '\n')
        answer_options  = re.sub(r'/\s*$', '', answer_options)
        
        ## Filter Question String: Remove Correct Answer and Answer Options
        answer_option_index = string_question.find("answer option")
        if answer_option_index != -1:
            string_question = string_question[:answer_option_index]
        else:
            correct_answer_index = string_question.find("correct answer")
            if correct_answer_index != -1:
                string_question = string_question[:correct_answer_index]

        ## convert correct answer for FITB question
        if type_fitb == True:
            correct_answer = await self._convert_fitb_answer(correct_answer)

        ## Filter URL to capture the right pattern
        string_urls = await self._filter_and_correct_urls(string_urls)
        string_urls_solutions = await self._filter_and_correct_urls(string_urls_solutions)

        ## Return Unique Items in String URLS
        string_urls = list(set(string_urls))

        ## check for images in answer options, if exist, we don't need to clean using BeautifulSoup as it might remove the image tags, instead we will extract the image urls and use them as answer options.
        answer_options_list = img_url_pattern.findall(answer_options)

        if answer_options_list:
            answer_options = answer_options_list
            for answer_option in answer_options:
                if answer_option in correct_answer:
                    correct_answer = answer_option
            
            ## filter out string urls if exist in answer_options
            string_urls = [
                url for url in string_urls
                if not any(opt in url for opt in answer_options)
            ]

        ## if no images detected, clean HTML tags from answer options, correct answer
        else:
            soup = BeautifulSoup(answer_options, 'html.parser')
            answer_options = soup.get_text()

            ## finally, make it into a list by separating $$
            answer_options = [part.strip() for part in answer_options.split('$$') if part.strip()]

            soup = BeautifulSoup(correct_answer, 'html.parser')
            correct_answer = soup.get_text()

        ## if solution contains images
        solution_images = img_url_pattern.findall(solution)

        if not solution_images:
            solution_images = string_urls_solutions

        if solution_images:
            ## if solution images exist, remove from string_urls
            string_urls = [
                url for url in string_urls
                if not any(opt in url for opt in solution_images)
            ]
        else:
            ## if solution has no images, we can clean
            soup = BeautifulSoup(solution, 'html.parser')
            solution = soup.get_text()

        ## Final Clean string_question
        if 'Article / Diagram from main question' not in string_question:
            ## if the question is not a sub question
            string_question = re.sub(r'question\s*:\s*', '', string_question).strip()
            
        string_question = re.sub('<br/>', '', string_question).strip()

        return type_fitb, type_mcq, string_question, string_urls, answer_options, correct_answer, mark, solution, solution_images

    def _is_image_filename(self, answer: str) -> bool:
        if not isinstance(answer, str):
            return False
        answer = answer.strip()
        # Pattern 1: Contains "file" (e.g., "1741849489267-file")
        if '-file' in answer.lower():
            return True
        # Pattern 2: Standard image extensions (e.g., "image.jpg")
        pattern = r'^.+\.(png|jpg|jpeg|gif|webp)$'
        return bool(re.match(pattern, answer, re.IGNORECASE))

    def _construct_student_image_url(self, filename: str) -> str:
        filename = filename.strip()
        # Fix URLs with spaces after the colon
        filename = re.sub(r'^(https?):\s*//', r'\1://', filename)
        # If already a full URL, return as is
        if filename.startswith('http://') or filename.startswith('https://'):
            return filename
        # Otherwise, prepend the base URL
        return f"{STUDENT_UPLOAD_BASE_URL}{filename}"

    async def preprocess_student_answer(
        self, 
        student_answer: Union[str, List[str]]
    ) -> Tuple[Union[str, List[str]], Optional[Union[str, List[str]]]]:
        """ Preprocess student answer to filter any unnecessary patterns that may affect AI behavior.
        Example:
        - Limit consecutive newlines to a maximum of 3.
        - Limit consecutive \left( and \right) tags to a maximum of 2. (some answers might abuse these tags)
        """
        
        async def process(s: str) -> Tuple[str, Optional[str]]:
            try:
                if self._is_image_filename(s):
                    return s, self._construct_student_image_url(s)
                
                s = json.loads(s)
                
                if isinstance(s, int) or isinstance(s, float):
                    s = str(s)

                if isinstance(s, list):
                    s = str(s)
            
            except Exception as e:
                print(f"Preprocess Student Answer Error: {e}")
                pass

            ## If student answers contains too many consecutive newlines, can cause AI to behave incorrectly. Max is \n\n\n
            s = re.sub(r'\n{4,}', '\n\n\n', s)
            ## If student answers contains too many \\left and \\right tags, delete also
            s = re.sub(r'(\\left\()+', lambda m: '\\left(' * min(2, m.group(0).count('\\left(')), s)
            s = re.sub(r'(\\right\))+', lambda m: '\\right)' * min(2, m.group(0).count('\\right)')), s)
            s = re.sub(r'(<)+', lambda m: '<' * min(2, len(m.group(0))), s)

            return s, None

        if isinstance(student_answer, str):
            return await process(student_answer)
        
        elif isinstance(student_answer, list):
            # Ensure each answer is processed
            results = [await process(ans) for ans in student_answer]
            answers = [r[0] for r in results]
            image_urls = [r[1] for r in results]
            
            if all(url is None for url in image_urls):
                return answers, None
            
            return answers, image_urls
            # return [await process(ans) for ans in student_answer]
        else:
            return student_answer, None
        
    async def extract_model_answers_fitb(
        self, 
        correct_answer: str,
        fitb_index: int   
    ) -> List[str]:
        pattern = rf"blank {fitb_index}\s*:\s*([^/]+)"
    
        # Find all matches
        matches = re.findall(pattern, correct_answer)
        
        # Clean up whitespace and return list
        return [match.strip() for match in matches]
    
    async def extract_model_answers(
        self,
        correct_answer: str
    ) -> List[str]:
        raw_list = correct_answer.split('/')
        clean_list = [answer.strip() for answer in raw_list if answer.strip()]

        return clean_list
    
    async def clean_student_answer(
        self,
        student_answer: str,
    ) -> str:
        s = unicodedata.normalize("NFC", student_answer)
        s = s.casefold()
        s = re.sub(r'[^\w\s]|_', '', s, flags=re.UNICODE)

        s = s.strip()
        s = re.sub(r'\s+', ' ', s)
        return s

    async def analyze_typos(
        self,
        correct_answer: str,
        student_answer: str 
    ) -> dict:
        def normalize(s):
            s = s.strip()
            s = re.sub(r'\s+', ' ', s)
            return s.lower()
        # clean both correct answer and student answer from excess whitespace at the end (if any)
        correct_answer = normalize(correct_answer)
        student_answer = normalize(student_answer)

        # Process at character level
        out = jiwer.process_characters(correct_answer, student_answer)
        
        # Extract the counts
        hits = out.hits
        substitutions = out.substitutions
        deletions = out.deletions
        insertions = out.insertions
        
        return {
            "total_errors": substitutions + deletions + insertions,
            "breakdown": {
                "insertions": insertions,
                "substitutions": substitutions,
                "deletions": deletions
            }
        }

    async def reorder_fitb_index(
        self,
        fitb_index: Union[int, List[int]]
    ) -> List[int]:
        try:
            if not fitb_index:
                return fitb_index  # handle empty list
            if fitb_index[0] == 1:
                return fitb_index  # already starts with 1, no change needed
            diff = fitb_index[0] - 1
            return [x - diff for x in fitb_index]
        except:
            return fitb_index  # in case of any error, return original
        
    async def constructQuestionContext(self, api_key) -> QuestionContext:
        ## initialization
        reqs = self.reqs
        rubric_transformer = RubricTransformer(reqs = reqs)
        question_json = json.loads(reqs.question)
        audio_transcriptions = ""

        student_answer, student_image_urls = await self.preprocess_student_answer(reqs.student_answer)
        rubric_context = rubric_transformer.constructRubricContext()

        ## obtain question properties from JSON structure of the question payload
        type_fitb, type_mcq, string_question, string_urls, answer_options, correct_answer, mark, solution, solution_images = await self.extract_question_obj(question_json)

        ## If Audio Exists, We Should Use STT to Extract the Text
        audio_urls = re.findall(r'\S+\.(?:mp3|wav|ogg)', string_question)
        
        if not solution:
            solution = solution_images

        ## if marking_schema is "per_question", there might be chance that fitb_index does not start with 1 (in case of there are multiple questions inside the worksheet)
        ## in that case, we need to adjust the fitb_index accordingly
        if type_fitb and reqs.marking_schema == "per_question":
            reqs.fitb_index = await self.reorder_fitb_index(reqs.fitb_index)
        
        print(f"Current FITB Index: {reqs.fitb_index}")

        ## Construct mark_context object
        mark_context = QuestionMarkContext(
            full_mark = mark if type_fitb else reqs.full_mark,
            step = str(reqs.step),
            type_fitb = type_fitb,
            fitb_index = reqs.fitb_index,
            num_blanks = 0,
            unique_blanks = 0
        )

        ## if question type if FITB, 
        if type_fitb == True:
            correct_answer_filtered, num_blanks, unique_blanks = await self.extract_blank_section_dispatch(correct_answer, reqs.fitb_index)
            mark_context.num_blanks = num_blanks
            mark_context.unique_blanks = unique_blanks

            ## based on the marking scheme for FITB chosen, the full mark will be different
            if reqs.marking_schema in ["per_question_zero_or_full"]:
                mark_context.full_mark = float(mark) / unique_blanks
            
            if reqs.marking_schema in ["advanced"]:
                mark_context.full_mark = 1

        else:
            pass

        ## for non FITB, answer_options == correct_answer
        provided_correct_answer = ""
        if type_fitb:
            provided_correct_answer = correct_answer_filtered
        elif type_mcq:
            provided_correct_answer = correct_answer
        else:
            provided_correct_answer = answer_options
        
        ## construct answer_context object
        answer_context = AnswerContext(
            correct_answer = provided_correct_answer, 
            answer_options = answer_options,
            original_correct_answer = correct_answer if type_fitb else "",
            pas_answer = reqs.answer_pas,
            student_answer = student_answer,
            student_image_urls = student_image_urls,
            solution = solution
        )

        ## construct question_context object
        question_context = QuestionContext(
            question = string_question,
            string_urls = string_urls,
            incorrect_flags = reqs.incorrect_flags,
            marking_note = reqs.marking_note,
            subject = reqs.subject,
            language = reqs.language,
            mark_context = mark_context,
            answer_context = answer_context,
            rubric_context = rubric_context
        )
        
        return question_context

