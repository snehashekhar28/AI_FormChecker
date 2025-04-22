// app/(tabs)/upload.tsx
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
  Platform,
} from 'react-native';
import { Video } from 'expo-av';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';

export default function UploadScreen() {
  const router = useRouter();
  const [videoUri, setVideoUri] = useState<string| null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // ask permissions once
  useEffect(() => {
    (async () => {
      if (Platform.OS !== 'web') {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert('Permission required', 'Please allow photo library access.');
        }
      }
    })();
  }, []);

  // pick-and-upload in one go
  const pickAndUpload = async () => {
    try {
      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 0.8,
      });
      if (res.cancelled) return;
      setVideoUri(res.uri);
      setUploading(true);
      setUploadProgress(0);

      // stub upload progress
      let pct = 0;
      const iv = setInterval(() => {
        pct += 25;
        setUploadProgress(pct);
      }, 300);

      // after ~1.2s, go to Loading
      setTimeout(() => {
        clearInterval(iv);
        setUploading(false);
        router.push({
          pathname: '/loading',
          params: { videoUri: res.uri },
        });
      }, 1200);
    } catch (err) {
      console.error(err);
      Alert.alert('Error', 'Could not select video.');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Upload Your Workout Video</Text>

      <TouchableOpacity
        style={[styles.button, uploading && { opacity: 0.6 }]}
        onPress={pickAndUpload}
        disabled={uploading}
      >
        <Text style={styles.buttonText}>
          {uploading ? `Uploading ${uploadProgress}%` : 'Select & Analyze Video'}
        </Text>
      </TouchableOpacity>

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
    </View>
  );
}

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
    backgroundColor: '#FF5A5A',
    paddingVertical: 14,
    paddingHorizontal: 30,
    borderRadius: 8,
    marginBottom: 15,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    textAlign: 'center',
  },
  videoContainer: {
    width: '100%',
    height: 200,
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#2C2C2C',
    marginTop: 20,
  },
  video: {
    width: '100%',
    height: '100%',
  },
});
