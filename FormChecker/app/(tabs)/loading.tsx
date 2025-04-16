import React, { useEffect, useRef, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  ScrollView,
} from 'react-native';
import { Video } from 'expo-av';
import BottomSheet from '@gorhom/bottom-sheet';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

const LoadingScreen: React.FC = () => {
  const bottomSheetRef = useRef<BottomSheet>(null);
  const [resultsReady, setResultsReady] = useState<boolean>(false);

  // Example values for score and LLM feedback
  const score = 8.5;
  const feedback =
    'Your pose was overall very good—keep working on your balance and core strength for even better results. Remember to focus on smooth transitions between movements.';

  // Define snap points: initially closed, then expands to 80% height.
  const snapPoints = useMemo(() => ['1%', '80%'], []);

  // Simulate processing delay (e.g., running MediaPipe and model scoring)
  useEffect(() => {
    const timer = setTimeout(() => {
      setResultsReady(true);
      // Expand the bottom sheet when results are ready
      bottomSheetRef.current?.expand();
    }, 5000); // Adjust delay as needed
    return () => clearTimeout(timer);
  }, []);

  // Render the bottom sheet content
  const renderBottomSheetContent = () => (
    <View style={styles.sheetContent}>
      <View style={styles.sheetHeader}>
        <View style={styles.handle} />
      </View>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.sheetTitle}>Exercise Analysis</Text>
        <Text style={styles.sheetScore}>Score: {score.toFixed(1)}/10</Text>
        <Text style={styles.sheetFeedback}>{feedback}</Text>
      </ScrollView>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Background video placeholder */}
      <Video
        source={require('../../assets/squat_7.mp4')} // Adjust your asset path accordingly
        rate={1.0}
        volume={1.0}
        isMuted
        resizeMode="cover"
        shouldPlay
        isLooping
        style={styles.backgroundVideo}
      />

      {/* Loading overlay */}
      {!resultsReady && (
        <View style={styles.loadingOverlay}>
          <Text style={styles.loadingText}>Analyzing your video…</Text>
        </View>
      )}

      {/* @gorhom/bottom-sheet */}
      <BottomSheet
        ref={bottomSheetRef}
        index={resultsReady ? 1 : 0}
        snapPoints={snapPoints}
        enablePanDownToClose
        style={styles.bottomSheet}
      >
        {renderBottomSheetContent()}
      </BottomSheet>
    </View>
  );
};

export default LoadingScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backgroundVideo: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#FFFFFF',
    fontSize: 18,
  },
  bottomSheet: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -3 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  sheetContent: {
    backgroundColor: '#FFFFFF',
    height: SCREEN_HEIGHT * 0.8,
    padding: 16,
  },
  sheetHeader: {
    alignItems: 'center',
    paddingVertical: 10,
  },
  handle: {
    width: 40,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#ccc',
  },
  scrollContent: {
    paddingBottom: 20,
  },
  sheetTitle: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 10,
    color: '#333',
  },
  sheetScore: {
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 10,
    color: '#333',
  },
  sheetFeedback: {
    fontSize: 16,
    lineHeight: 22,
    color: '#555',
  },
});
