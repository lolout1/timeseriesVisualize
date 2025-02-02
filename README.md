# timeseriesVisualize
Visualizing time series analysis

We are smoothening 3d skeleton joint data using kalman smoothening per axis. We use this 3D 32 joint skeleton data as an additional modality to train our teacher model for fall detection. This smoothened joint data should ideally remove the noisy signal characteristics in the signal while preserving the characteristics required to detect falls( due to human error when logging, inaccuracuate sensors, etc) and result in a more knowledgable and accurate teacher model in the context of fall detection. 
