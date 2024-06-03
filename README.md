# NLP2
1. The different experiments (ft_s, ft_l, ft_lmhead and rq3) are included in their respective results folder
2. All json result files can be found in the ```results``` folder.
3. Visualisation of these restults can be made with ```visualisations.ipynb```
   
## Fine-tuning

## Few-shot (FS) learning
1. We randomly choose train examples from the Provo dataset, which can be done with ```build_prompts.py```
2. Use the nlp.job file to generate continuations with few-shot prompting and Gemma ```nlp.job```
3. ```postprocess.py``` can be used to extract the continuations from the output file
4. ```scores.py``` can be used to calculate the TVD values for the different FS variants.
