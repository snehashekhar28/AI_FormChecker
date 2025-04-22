// app/(tabs)/loading.tsx
import React, { useEffect } from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';
import { Video } from 'expo-av';
import { useRouter, useLocalSearchParams } from 'expo-router';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

export default function LoadingScreen() {
  const router = useRouter();
  const { videoUri } = useLocalSearchParams<{ videoUri: string }>();
  
  useEffect(() => {
    const timer = setTimeout(() => {
      // Replace with real analysis results when ready
      router.replace({
        pathname: '/results',
        params: {
          score: '7.2',
          feedback: 'Great effort! Keep your back straighter during squats to protect your spine.',
          videoUri,
          workoutType: 'Squat',
        },
      });
    }, 3000); // 3 second simulated processing

    return () => clearTimeout(timer);
  }, [router, videoUri]);

  return (
    <View style={styles.container}>
      <Video
        source={{ uri: videoUri }}
        style={styles.video}
        shouldPlay
        isLooping
        isMuted
        resizeMode="cover"
      />
      <View style={styles.overlay}>
        <Text style={styles.text}>Analyzing your workout…</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000', // fallback
  },
  video: {
    width: SCREEN_WIDTH,
    height: SCREEN_HEIGHT,
    position: 'absolute',
    top: 0,
    left: 0,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  text: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: '500',
  },
});
