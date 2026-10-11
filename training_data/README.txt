TRAINING DATA FOLDER
====================

Put your training text files here. The model learns from whatever you feed it.

SUGGESTED ORDER:
1. simple_qa.txt       — Basic Q&A pairs (teaches conversation structure)
2. topic_articles.txt  — Articles on a subject you care about (adds knowledge)
3. personal_style.txt  — Your own writing (makes it sound like you)

FORMAT:
- Plain text (.txt) files work best
- One file per topic or category
- For Q&A: write each pair as "Q: ..." and "A: ..." on separate lines
- For articles/stories: just paste the text normally

HOW TO USE:
1. Create or download .txt files into this folder
2. Open the Lattice-Grow dashboard
3. Go to the Training tab
4. Upload files from this folder or paste text directly
5. Click "Start Training"

TIPS:
- Start small (20-30 Q&A pairs) before adding large datasets
- Small, focused training sessions build better experts than one giant dump
- You can also use the Autonomous Web Learning tab to crawl websites
  once your model has at least a few experts activated

EXAMPLE Q&A FORMAT:
Q: What is your name?
A: My name is Nyu.

Q: What can you do?
A: I can answer questions and have conversations with you.

Q: How were you made?
A: I was built using the Lattice-Grow architecture on a local laptop.
