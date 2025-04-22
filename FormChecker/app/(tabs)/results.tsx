// app/(tabs)/results.tsx
import React, { useMemo } from 'react';
import {
  ScrollView,
  View,
  Text,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
} from 'react-native';
import { Video } from 'expo-av';
import { LinearGradient } from 'expo-linear-gradient';
import { useRouter, useLocalSearchParams } from 'expo-router';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const BAR_WIDTH = SCREEN_WIDTH * 0.9;
const BAR_HEIGHT = 20;
const HANDLE_SIZE = 24;

type Params = {
  score: string;
  feedback: string;
  videoUri: string;
  workoutType: string;
};

export default function ResultsScreen() {
  const router = useRouter();
  const { score = '0', feedback = '', videoUri = '', workoutType = '' } =
    useLocalSearchParams<Params>();

  const numericScore = Number(score);
  const handleLeft = useMemo(
    () => (numericScore / 10) * (BAR_WIDTH - HANDLE_SIZE),
    [numericScore]
  );

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.header}>Your score for{'\n'}this workout</Text>
      <Text style={styles.scoreText}>{numericScore}/10</Text>

      <View style={styles.barContainer}>
        <LinearGradient
          colors={['#F56565', '#ED8936', '#48BB78']}
          start={[0, 0]}
          end={[1, 0]}
          style={styles.gradientBar}
        />
        <View style={[styles.handle, { left: handleLeft }]} />
      </View>

      <Video
        source={{ uri: videoUri }}
        style={styles.video}
        useNativeControls
        resizeMode="contain"
      />

      <Text style={styles.workoutType}>Workout type: {workoutType}</Text>

      <Text style={styles.subheader}>Improvement suggestions</Text>
      <Text style={styles.feedback}>
        {feedback}{' '}
        <Text style={styles.viewMore}>View more</Text>
      </Text>

      <TouchableOpacity
        style={styles.recordButton}
        onPress={() => router.replace('/upload')}
      >
        <Text style={styles.recordButtonText}>Record new workout</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#1C1C1C', padding: 16 },
  header:   { color: '#FFF', fontSize: 28, fontWeight: '700', textAlign: 'center', marginTop: 16 },
  scoreText:{ color: '#FF5A5A', fontSize: 48, fontWeight: '700', textAlign: 'center', marginBottom: 16 },
  barContainer: {
    width: BAR_WIDTH,
    height: BAR_HEIGHT + HANDLE_SIZE / 2,
    alignSelf: 'center',
    marginBottom: 24,
    justifyContent: 'center',
  },
  gradientBar: {
    width: BAR_WIDTH,
    height: BAR_HEIGHT,
    borderRadius: BAR_HEIGHT / 2,
  },
  handle: {
    position: 'absolute',
    top: -(HANDLE_SIZE - BAR_HEIGHT) / 2,
    width: HANDLE_SIZE,
    height: HANDLE_SIZE,
    borderRadius: HANDLE_SIZE / 2,
    backgroundColor: '#1C1C1C',
    borderWidth: 2,
    borderColor: '#FFF',
  },
  video: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    marginBottom: 12,
    backgroundColor: '#000',
  },
  workoutType:   { color: '#FFF', fontSize: 16, marginBottom: 12 },
  subheader:     { color: '#FF5A5A', fontSize: 20, fontWeight: '600', marginBottom: 6 },
  feedback:      { color: '#FFF', fontSize: 16, lineHeight: 22, marginBottom: 24 },
  viewMore:      { color: '#BBB', textDecorationLine: 'underline', fontSize: 14 },
  recordButton: {
    backgroundColor: '#FF5A5A',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 32,
  },
  recordButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: '700',
  },
});
