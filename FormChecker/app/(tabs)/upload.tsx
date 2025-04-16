import React, { useState } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  StyleSheet, 
  ActivityIndicator, 
  Alert 
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { Video } from 'expo-av';
import axios from 'axios';

const UploadScreen: React.FC = () => {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [videoType, setVideoType] = useState<string | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);

  // Function to pick a video file
  const pickVideo = async (): Promise<void> => {
    try {
      const result = await DocumentPicker.getDocumentAsync({ type: 'video/*' });
      // Instead of checking result.type, check if the selection wasn't cancelled.
      if (!result.cancelled) {
        setVideoUri(result.uri);
        // If mimeType is not provided, default to 'video/mp4'
        setVideoType(result.mimeType || 'video/mp4');
      }
    } catch (error) {
      Alert.alert('Error', 'An error occurred while picking the video.');
      console.error(error);
    }
  };

  // Function to upload the video to the backend
  const uploadVideo = async (): Promise<void> => {
    if (!videoUri) {
      Alert.alert('No Video Selected', 'Please select a video before uploading.');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('video', {
      uri: videoUri,
      type: videoType || 'video/mp4',
      name: 'upload_video.mp4', // Consider generating a unique name
    } as any); // Casting to any to work around TS issues with FormData file types

    try {
      // Replace 'YOUR_API_URL/api/upload' with your actual backend endpoint
      const response = await axios.post('YOUR_API_URL/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent: ProgressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        },
      });
      console.log('Upload success:', response.data);
      Alert.alert('Success', 'Video uploaded successfully.');
      // Navigate to a processing screen or update UI state here if needed
    } catch (error) {
      console.error('Upload failed:', error);
      Alert.alert('Upload Error', 'Failed to upload the video. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Upload Your Workout Video</Text>
      
      {/* Button to select a video */}
      <TouchableOpacity 
        style={styles.button} 
        onPress={pickVideo} 
        disabled={uploading}
      >
        <Text style={styles.buttonText}>Select Video</Text>
      </TouchableOpacity>

      {/* Video preview */}
      {videoUri && (
        <View style={styles.videoContainer}>
          <Video
            source={{ uri: videoUri }}
            useNativeControls
            resizeMode="contain"
            style={styles.video}
          />
        </View>
      )}

      {/* Upload progress indicator and upload button */}
      {uploading ? (
        <View style={styles.progressContainer}>
          <ActivityIndicator size="large" color="#e9535f" />
          <Text style={styles.progressText}>Uploading {uploadProgress}%</Text>
        </View>
      ) : (
        videoUri && (
          <TouchableOpacity 
            style={styles.uploadButton} 
            onPress={uploadVideo}
          >
            <Text style={styles.buttonText}>Upload Video</Text>
          </TouchableOpacity>
        )
      )}

      {/* Instructions */}
      <Text style={styles.instructions}>
        Tap to select a workout video to be analyzed.
      </Text>
    </View>
  );
};

export default UploadScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1C1C1C',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#e9535f',
    paddingVertical: 12,
    paddingHorizontal: 25,
    borderRadius: 8,
    marginBottom: 15,
  },
  uploadButton: {
    backgroundColor: '#e9535f',
    paddingVertical: 12,
    paddingHorizontal: 25,
    borderRadius: 8,
    marginTop: 15,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
  },
  videoContainer: {
    width: '100%',
    height: 200,
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#2C2C2C',
    marginBottom: 15,
  },
  video: {
    width: '100%',
    height: '100%',
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 15,
  },
  progressText: {
    color: '#FFFFFF',
    marginLeft: 10,
    fontSize: 16,
  },
  instructions: {
    color: '#D9D9D9',
    fontSize: 14,
    textAlign: 'center',
    marginTop: 20,
  },
});
