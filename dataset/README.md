# Audio-IMU multimodal cough dataset using wearables

[https://doi.org/10.5061/dryad.mkkwh717r](https://doi.org/10.5061/dryad.mkkwh717r)

Cough detection is essential for long-term respiratory illness monitoring, but clinical methods are not feasible for home use. Wearable devices offer a convenient alternative, but challenges include data limitation and accurately detecting coughs in real-world environments, where audio quality may be compromised by background noise. This multimodal dataset, collected in a controlled lab setting, includes IMU and audio data captured using wearable devices. It was designed to support the development of an accessible and effective cough detection system. The dataset documentation includes details on sensor arrangement, data collection protocol, and processing methods.

Our analysis reveals that integrating transfer learning, multimodal approaches, and out-of-distribution (OOD) detection significantly enhances system performance. Without OOD inputs, the model achieves accuracies of 92.59% in the in-subject setting and 90.79% in the cross-subject setting. Even with OOD inputs, the system maintains high accuracies of 91.97% and 90.31%, respectively, by employing OOD detection techniques, despite the OOD inputs being double the number of in-distribution (ID) inputs. These results are promising for developing a more efficient and user-friendly cough and speech detection system suitable for wearable technology.

## Description of the data and file structure

This dataset comprises recordings from 13 participants, collected as part of a study approved by NC State University IRB Protocol 25003. The participants, who are student volunteers of similar age and health condition, engaged in a series of physical activities, including sitting, walking, running, and transitions between these states, each lasting approximately two minutes with 30-second resting intervals between transitions.

### Data Collection Overview:

* **Audio Recording:** Two chest-mounted microphones were used to capture audio data. One microphone faced away from the participant (out-microphone), and the other faced toward the participant (in-microphone). The microphones were housed in a custom-designed enclosure and sourced from Tozo T10 Bluetooth earbuds with the speaker circuit disconnected.
* **IMU Data:** The participant's movement was recorded using a MetaMotionS r1 sensor from Mbientlab, mounted on the chest to capture 9-axis Inertial Measurement Unit (IMU) data.
  Synchronization:

At the start of each recording session, participants clapped three times, which serves as a synchronization point across different data modalities. These claps are clearly identifiable in both the audio and IMU signals, allowing for precise alignment of the datasets.

### Data Labeling:

The audio data was labeled using the open-source tool [Audino](https://github.com/midas-research/audino). The sounds recorded were categorized into several classes:

* Cough
* Speech
* Sneeze
* Deep Breath
* Groan
* Laugh
* Speech (far): Speech from individuals near the participant.
* Other Sounds: Environmental noises and periods of silence.

### File Structure:

**Audio Files:**

The recordings are captured from two chest-mounted microphones and are labeled by participant ID and activity type.
Files from the in-microphone are named with `_In.wav`, while those from the out-microphone are named with `_Out.wav`.
Note: Some audio recordings from the in-microphone may be missing.

**IMU Data Files:**

These files contain 9-axis movement data, synchronized with the audio recordings.
The files are saved under folder named with participant ID and activity type and are saved in ".csv" format.

**Labels Files:**

* DataAnnotation.json: This file contains annotated sound events with timestamps, linking them to the corresponding audio and IMU files.
* sync_time.txt: This file contains the timestamps used for synchronization between the audio and IMU data.

**Potential Use:**

Users can employ this dataset to develop and validate algorithms for cough detection and other sound recognition tasks in wearable technology. The clear structure and detailed annotations make it suitable for those new to the field, as well as experts aiming to advance cough detection technology.

**Organization:**

The dataset is organized in following structure. The folders are named by subject ID and trial number.

MultimodalCoughDataset.zip
\|- DataAnnotation.json
\|- 005/
\|  |- Trial_1_No_Talking/
\|  |   |- 005_Talking_In.wav
\|  |   |- 005_Talking_Out.wav
\|  |   |- Accelerometer.csv
\|  |   |- Gyroscope.csv
\|  |   |- Magnetometer.csv
\|  |- Trial_2_Talking/
\|  |- Trial_3_Nonverbal/
\|  |- sync_time.txt
\|- 006/
\|  |- Trial_1_No_Talking/
\|  |- Trial_2_Talking/
\|  |- Trial_3_Nonverbal/
\|  |- sync_time.txt

## Data Analysis

We developed a multimodal cough detection system utilizing a pre-trained MobileNet for the audio modality and a customized CNN for the IMU modality. Our analysis shows that multimodal models outperform single-modal models in both in-subject and cross-subject settings, particularly in cough detection. The addition of the IMU modality significantly enhances detection performance, especially at lower audio frequencies. Moreover, transfer learning substantially improves prediction accuracy, and the use of an enhanced balanced dataset, along with a weighted multi-loss function, further boosts the effectiveness of multimodal modeling. The optimized multimodal model remains stable across various window sizes and audio frequencies, unlike the single-modal model, which is sensitive to these factors.

## Related Works

The details of data collection and processing can be found in our work submited to 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBS) (In print). Through our analyses, we demonstrate that combining transfer learning, a multimodal approach, and OOD detection techniques significantly improves system performance. Without OOD inputs, the system shows high accuracies of 92.59% in the in-subject setting and 90.79% in the cross-subject setting. With OOD inputs, it still maintains overall accuracies of 91.97% and 90.31% in these respective settings by incorporating OOD detection, despite the number of OOD inputs being twice that of In-Distribution (ID) inputs. This research are promising towards a more efficient, user-friendly cough and speech detection method suitable for wearable devices.

## Code/Software

The code for data synchronization and related paper can be found in [Multimodal Cough Detection with Out-of-Distribution Detection](https://github.com/ARoS-NCSU/OOD-Multimodal-CoughDet) and [Robust Multimodal Cough Detection with Optimized Out-of-Distribution Detection for Wearables](https://github.com/ARoS-NCSU/Optimized-OOD-Multimodal-CoughDet/tree/main)