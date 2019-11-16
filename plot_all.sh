

for i in plot_running_mAP plot_action_histograms plot_training plot_scatters plot_summaries plot_size_density; 
  do
    python -m "scripts.$i"
  done
