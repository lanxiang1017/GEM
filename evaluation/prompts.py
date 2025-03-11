arean_eval_prompt = """Evaluate the quality of a model's response to an ECG-related question by comparing it with a given ground truth answer. Focus on three aspects: accuracy, completeness, and instruction adherence. Be precise and objective, especially when identifying errors in medical terminology. Do not let the response length affect your evaluation.

Evaluation Criteria:
1. Accuracy (0-10):
How well does the model's response match the ground truth, particularly in ECG interpretation and diagnosis? This score emphasizes whether the key information is correct, such as the correct identification of waveforms, intervals, and clinical diagnoses.
- 10: Fully accurate, with correct ECG interpretation, terminology, and diagnosis.
- 5: Partially accurate, with some correct information but notable errors or omissions.
- 0: Largely inaccurate or misleading.

2. Completeness (0-10):
Does the response cover essential aspects of ECG interpretation (e.g., rhythm, axis, waveforms, clinical causes) mentioned in the ground truth? This score focuses on whether the answer is comprehensive and includes as much essential information as possible.
- 10: Comprehensive, covering all key details.
- 5: Partially complete, with important points missing.
- 0: Incomplete, lacking critical information.

3. Instruction Adherence (0-10):
Does the model follow the specific instructions in the question (e.g., listing features, suggesting a diagnosis)? This score focuses on how well the model follows the task instructions, regardless of the correctness of the answer.
- 10: Fully follows instructions.
- 5: Partially follows instructions, with some deviations.
- 0: Fails to follow instructions or provides an irrelevant response.

Please organize your output in a JSON format of accuracy, completeness, and instruction adherence, with a brief explanation of each aspect. For example: {Accuracy: {Score: $SCORE$, Explanation: $EXPLANATION$}}"""


report_eval_prompt = """Evaluate the alignment and quality of a generated ECG report by comparing it to a ground truth clinician’s report. The evaluation will focus on three key aspects: Diagnosis, Waveform, and Rhythm. Use specific criteria for each aspect and be precise in comparing medical terminologies. Only focus on information present in the ground truth report, identifying any mistakes. Remain objective and do not let the response length affect your evaluation.

Evaluation Criteria:
1. Diagnosis (0-10):
Assess how well the generated ECG report matches the clinical diagnoses in the ground truth report. Focus on conditions like conduction disturbances, ischemia, hypertrophy, and other abnormalities as presented in the ground truth report.
- 10: All key diagnoses are correctly identified with no errors or omissions.
- 5: Partially accurate, with some diagnoses identified correctly but key conditions missing or incorrect.
- 0: Fails to identify key diagnoses, with multiple critical errors.

2. Waveform (0-10):
Evaluate the accuracy and quality of the ECG waveform morphology in the generated report compared to the ground truth. Focus on abnormalities in P-wave, QRS complex, ST changes, T-wave, and intervals (PR, QT), ensuring waveform morphology is consistent with the ground truth.
- 10: All waveform abnormalities are correctly identified without errors.
- 5: Some waveform abnormalities are identified, but key issues are missed or misinterpreted.
- 0: Fails to identify key waveform abnormalities, with multiple critical errors.

3. Rhythm (0-10):
Assess the accuracy and clarity of rhythm interpretation in the generated report. Focus on identifying and describing normal and abnormal rhythms (e.g., sinus rhythm, atrial fibrillation, ventricular tachycardia) as presented in the ground truth report.
- 10: Rhythm interpretation is fully accurate and clearly described.
- 5: Rhythm interpretation is partially accurate but contains notable errors or omissions.
- 0: Rhythm interpretation is largely incorrect, with critical errors.

Please organize your output in a JSON format of diagnosis, form and rhythm, with a brief explanation of each aspect. For example: {Diagnosis: {Score: $SCORE$, Explanation: $EXPLANATION$}}"""