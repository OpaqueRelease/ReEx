
## generic run to show functionality. Run from this folder!
python generate_benchmark_jobs.py | parallel >> ../benchmark_results/first_run.txt

# python reex

# python reex --reasoner reex-lgg --explanation_method class-ranking
# python reex --reasoner reex-lgg --explanation_method shap

# python reex --reasoner hedwig --explanation_method class-ranking
# python reex --reasoner hedwig --explanation_method shap

