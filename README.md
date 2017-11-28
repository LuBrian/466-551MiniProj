# 466MiniProj
CMPUT 466 Mini project
By: Fred Han (G) <xuefei1@ualberta.ca>, Brian Lu (UG) <blu11@ualberta.ca>
## Problem
- For our CMPUT 466/551 project, we plan to investigate the duplicate question problem. As the
name suggest, this problem investigates whether two questions in English are in fact, asking the
same thing. For example, questions like “Why should I take a machine learning course” and
“What are the reasons to take a machine learning course” are duplicate questions. For question
answering sites such as StackOverflow or Quora, detecting duplicate questions are essential as
it can help clean up their question servers.
- This is a binary classification problem, a pair of questions can only be duplicates of each other
or non-duplicates of each other.

- [Complete Draft](https://github.com/LuBrian/466MiniProj/blob/master/466_Mini_Project_draft.pdf)

Notes: 
- The feature dictionary we used is called "glove.6B.50d.txt" (171.4MB), it is too large to upload, you can obtain it from https://nlp.stanford.edu/projects/glove/ by downloading the "glove.6B.zip", we rename it as "glove_6B_50d_w2v.txt" in our project.
- According to large number of features on our data, the full runs with all parameters would approximately take 1 day.
