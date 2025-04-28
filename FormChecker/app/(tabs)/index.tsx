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
import { useVideoPlayer, VideoView } from 'expo-video';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import socket from '../utils/streaming';
import {resetLastAnalysis} from '../utils/streaming';

export default function UploadScreen() {
  const router = useRouter();
  const [videoUri, setVideoUri] = useState<string| null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [playerInstance, setPlayerInstance] = useState<any>(null);
  const player = useVideoPlayer({ uri: videoUri }, player => {
    player.loop = true;
    player.muted = true;
    player.play();
    setPlayerInstance(player);
  });

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

  useEffect(() => {
    if (!uploading && playerInstance) {
      playerInstance.pause();
    }
  }, [uploading, playerInstance]);

  // pick-and-upload in one go
  const pickAndUpload = async () => {
    try {
      resetLastAnalysis();
      setUploading(true);
      setUploadProgress(0);
      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: false,
        quality: 1,
      });
      console.log(res)
      if (res.canceled) {
        clearInterval(iv);
        setUploading(false);
        return;
      }
      setVideoUri(res.assets[0].uri);

      // stub upload progress
      let pct = 0;
      const iv = setInterval(() => {
        pct += 10;
        if (pct < 100) {
          setUploadProgress(pct);
        }
      }, 700);

      const form = new FormData();
      form.append('video', {
        uri: res.assets[0].uri,
        name: 'workout.mp4',
        type: 'video/mp4',
      } as any);

      const resp = await fetch('http://172.20.10.9:5000/upload', {
        method: 'POST',
        body: form,
        headers: {
          'X-Socket-ID': socket.id!,
        },
      });
      if (!resp.ok) throw new Error(await resp.text());

      const data = await resp.json();

      // go to loading after annotated video has been received
      if (data.processedVideoUrl) {
        clearInterval(iv);
        setUploadProgress(100);
        setUploading(false);
        router.push({
          pathname: '/loading',
          params: { videoUri: data.processedVideoUrl },
        });
      } else {
        Alert.alert('Server error', 'No processed video URL returned.');
      }
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

      {videoUri && player && (
        <View style={styles.videoContainer}>
          <VideoView style={styles.video} player={player} contentFit='contain' />
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
