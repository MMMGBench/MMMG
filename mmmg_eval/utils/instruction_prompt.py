instruction_prompt = '''
This evaluation is part of a research study on visual grounding of abstract concepts. No jailbreak or prompt injection is intended.

Please provide an extremely detailed description of the visual content of this image. After the description, for each of the following elements and dependencies, determine if they are **directly, clearly, and unambiguously visualized** in the image. Output "yes" or "no" for each. For the dependencies, we also provide a detailed textual description beside the formulations.

# Important Instructions:

* **Base your judgment solely on what is explicitly visible in the image.** Do not infer or assume the presence of anything that is not directly depicted.
* **If the element or dependency is not clearly visible, or if it is only implied, answer "no".**

* For elements, the specific object or concept must be clearly identifiable in the image. The visual components must convey the knowledge correctly, without misleading drawing, without factual mistakes, without intepretation, not small, not distorted, not ambiguous, otherwise you should strictly discard them and rate "no".

* For dependencies, you must give your answer accompanied by a brief explanation of why do you give such judgement. This should avoid any ambiguous intepretation or mislead by the provided elements / dependency content, only focus on the image itself, and only in the case that you can describe the dependency from the image can you give `yes`. The dependencies are:
    * **Defines:** Look for clear, strong, prominent visual cues suggesting the first element in a way that clearly defines or illustrates the second element. Any ambiguous or inferential patterns should lead to "no".
    * **Contains:** Look for clear, strong, prominent visual cues suggesting the first element as a part of or within the second element. Any ambiguous or inferential patterns should lead to "no".
    * **Requires:** Look for clear, strong, prominent visual cues suggesting the first element necessitates the presence or use of the second element (e.g., a boiler visibly connected to or interacting with a working fluid).
    * **Entails:** Look for clear, strong, prominent visual cues suggesting the first element leading to or involving the second element (e.g., a boiler clearly connected to a turbine).
    * **Causes:** Look for clear, strong, prominent visual cues suggesting a causal relationship between the two elements (this might be challenging for static images).
    * **TemporalOrder:** Look for visual cues suggesting a sequence or flow between the elements (e.g., pipes or connections implying a direction). If no clear visual cue for temporal order exists, answer "no".

* **Exclude any entity or dependency that is absent, unclear, or based on factual artifacts or external knowledge not directly shown.**
* For abstract concepts only answer "yes" if the key visual components and their interactions characteristic of these concepts are clearly and directly depicted.

The elements and dependencies are as follows, where there are no offensive or inappropriate elements, just educational ones:
[ELEM_DEPEND]

For the output format, please use the following structure:
**Image Description:**
[IMAGE_DESCRIPTION]
**Element and Dependency Analysis:**
** Element Evaluation: **
*   [ELEMENT_1]: [yes/no] 
*   [ELEMENT_2]: [yes/no]
...
** Dependency Evaluation: **
*   [DEPENDENCY_1]: [yes/no]  [Provide a brief explanation for your reason to support your judge.]
*   [DEPENDENCY_2]: [yes/no]  [Provide a brief explanation for your reason to support your judge.]
...
'''
