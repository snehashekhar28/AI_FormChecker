// app/(tabs)/history.tsx
import React, { useEffect, useState, useMemo } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  FlatList,
  Image,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import { useRouter } from 'expo-router';

interface Workout {
  id: string;
  type: string;
  date: string;        // ISO or display string
  score: number;       // 0–10
  thumbnail: string;   // URL or local asset
}

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function PastWorkoutsScreen() {
  const router = useRouter();
  const [workouts, setWorkouts] = useState<Workout[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState<'date' | 'score'>('date');

  // fetch from your backend here
  useEffect(() => {
    async function fetchWorkouts() {
      // TODO: replace this with real API call
      const data: Workout[] = [
        {
          id: '1',
          type: 'Squat',
          date: '2025-03-17',
          score: 7.8,
          thumbnail: 'https://example.com/thumb1.jpg',
        },
        {
          id: '2',
          type: 'Squat',
          date: '2025-04-07',
          score: 6,
          thumbnail: 'https://example.com/thumb2.jpg',
        },
        // …more dummy items
      ];
      // simulate network delay
      setTimeout(() => {
        setWorkouts(data);
        setLoading(false);
      }, 1000);
    }
    fetchWorkouts();
  }, []);

  // sort workouts based on sortBy
  const sorted = useMemo(() => {
    return [...workouts].sort((a, b) => {
      if (sortBy === 'date') {
        return new Date(b.date).getTime() - new Date(a.date).getTime();
      } else {
        return b.score - a.score;
      }
    });
  }, [workouts, sortBy]);

  const getCircleColor = (score: number) => {
    if (score >= 7) return '#48BB78';    // green
    if (score >= 4) return '#ED8936';    // orange
    return '#F56565';                    // red
  };

  const renderItem = ({ item }: { item: Workout }) => (
    <View style={styles.card}>
      <Image
        source={{ uri: item.thumbnail }}
        style={styles.thumbnail}
        resizeMode="cover"
      />
    <View style={[
    styles.scoreOuter,
    { borderColor: getCircleColor(item.score) },
]}>
    <View style={styles.scoreInner}>
        <Text style={styles.scoreInnerText}>
            {item.score.toFixed(1)}
        </Text>
    </View>
    </View>
      <View style={styles.info}>
        <Text style={styles.workoutType}>Workout: {item.type}</Text>
        <Text style={styles.workoutDate}>
          Date: {new Date(item.date).toLocaleDateString()}
        </Text>
        <TouchableOpacity
          style={styles.viewMoreButton}
          onPress={() => router.push(`/(tabs)/workout/${item.id}`)}
        >
          <Text style={styles.viewMoreText}>View more</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.header}>View past workouts</Text>

      {/* Sort By */}
      <View style={styles.sortRow}>
        <Text style={styles.sortLabel}>Sort by:</Text>
        <TouchableOpacity
          style={styles.sortButton}
          onPress={() =>
            setSortBy((prev) => (prev === 'date' ? 'score' : 'date'))
          }
        >
          <Text style={styles.sortButtonText}>
            {sortBy === 'date' ? 'Date ⌄' : 'Score ⌄'}
          </Text>
        </TouchableOpacity>
      </View>

      {loading ? (
        <ActivityIndicator size="large" color="#FF5A5A" style={{ flex: 1 }} />
      ) : (
        <FlatList
          data={sorted}
          keyExtractor={(item) => item.id}
          renderItem={renderItem}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
        />
      )}

      <TouchableOpacity
        style={styles.recordButton}
        onPress={() => router.push('/')}
      >
        <Text style={styles.recordButtonText}>Record new workout</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const CARD_HEIGHT = 100;
const THUMB_SIZE = CARD_HEIGHT - 16;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1C1C1C',
    paddingHorizontal: 16,
    paddingTop: 16,
  },
  header: {
    color: '#FFF',
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 12,
  },
  sortRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  sortLabel: {
    color: '#FFF',
    fontSize: 14,
    marginRight: 8,
  },
  sortButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#2C2C2C',
    borderRadius: 6,
  },
  sortButtonText: {
    color: '#FFF',
    fontSize: 14,
  },
  listContent: {
    paddingBottom: 16,
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    height: CARD_HEIGHT,
    marginBottom: 12,
    backgroundColor: '#2C2C2C',
    borderRadius: 8,
    overflow: 'hidden',
  },
  thumbnail: {
    width: THUMB_SIZE,
    height: THUMB_SIZE,
    borderRadius: 8,
  },
  scoreCircleContainer: {
    position: 'absolute',
    left: THUMB_SIZE + 16,
    width: 60,
    alignItems: 'center',
  },
  scoreCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  info: {
    marginLeft: THUMB_SIZE + 72,
    flex: 1,
    justifyContent: 'center',
  },
  workoutType: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '500',
  },
  workoutDate: {
    color: '#D9D9D9',
    fontSize: 12,
    marginBottom: 6,
  },
  viewMoreButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#FFF',
    borderRadius: 6,
  },
  viewMoreText: {
    color: '#1C1C1C',
    fontSize: 12,
    fontWeight: '500',
  },
  recordButton: {
    backgroundColor: '#FF5A5A',
    marginVertical: 16,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  recordButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: '700',
  },
    scoreOuter: {
      width: 60,
      height: 60,
      borderRadius: 30,
      borderWidth: 6,             // thickness of ring
      alignItems: 'center',
      justifyContent: 'center',
    },
  
    // Inner white circle
    scoreInner: {
      width: 44,                  // outer size minus 2*borderWidth (60 - 12)
      height: 44,
      borderRadius: 22,
      backgroundColor: '#FFFFFF',
      alignItems: 'center',
      justifyContent: 'center',
    },
  
    // Score text inside inner circle
    scoreInnerText: {
      color: '#FF5A5A',           // coral/red from your palette
      fontSize: 16,
      fontWeight: '700',
    },

});
