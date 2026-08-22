import os
import sys
sys.path.insert(0, os.path.abspath('Dastgah_Classifier_v5/src'))
from dastgah_v5.melodic_features import MelodicFeatureConfig, extract_track_feature
cfg = MelodicFeatureConfig(count_features=True)
res = extract_track_feature('Training_Data/Chahargah/Chahargah_0.wav', cfg, 'train', 42, 'Dastgah_Classifier_v5/data/cache')
print(res.shape)
