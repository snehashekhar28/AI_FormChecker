// app/(tabs)/_layout.tsx
import React, { useState, useEffect } from 'react';
import { Tabs } from 'expo-router';
import {
  Platform,
  View,
  Text,
  Image,
  StyleSheet,
  Dimensions,
} from 'react-native';
import { HapticTab } from '@/components/HapticTab';
import TabBarBackground from '@/components/ui/TabBarBackground';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

// explicit imports so Metro will error if file missing
import uploadIcon from '../../assets/images/barb.png';
import historyIcon from '../../assets/images/save.png';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function TabLayout() {
  const colorScheme = useColorScheme();
  const [showSplash, setShowSplash] = useState(true);

  // 3‑second splash
  useEffect(() => {
    const t = setTimeout(() => setShowSplash(false), 3000);
    return () => clearTimeout(t);
  }, []);

  if (showSplash) {
    return (
      <View style={styles.splashContainer}>
        <Image
          source={require('../../assets/logo.png')}
          style={styles.logo}
          resizeMode="contain"
        />
        <Text style={styles.splashText}>Welcome to Align</Text>
      </View>
    );
  }

  return (
    <Tabs
      // 1) start on Upload, not PastWorkouts
      initialRouteName="index"
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: false,

        // bottom placement everywhere
        tabBarPosition: 'bottom',

        // 2) only on web, hide the top‑tab indicator
        ...Platform.select({
          web: { tabBarIndicatorStyle: { height: 0 } },
        }),

        // dark bg + no top border
        tabBarStyle: [
          { backgroundColor: '#1C1C1C', borderTopWidth: 0 },
          Platform.select({
            ios: { position: 'absolute', height: 70, paddingBottom: 20 },
            default: { height: 60 },
          }),
        ],

        // white icons when active, light gray when inactive
        tabBarActiveTintColor: '#FF5A5A',
        tabBarInactiveTintColor: '#FFFFFF',

        // native-only haptics + background
        ...(Platform.OS !== 'web'
          ? {
              tabBarButton: HapticTab,
              tabBarBackground: TabBarBackground,
            }
          : {}),

        // spread them evenly
        tabBarItemStyle: {
          flex: 1,
          alignItems: 'center',
          justifyContent: 'center',
        },
      }}
    >
      {/* 1) Upload tab on the LEFT */}
      <Tabs.Screen
        name="index"
        options={{
          tabBarIcon: ({ color }) => (
            <Image
              source={uploadIcon}
              style={[styles.icon, { tintColor: color }]}
            />
          ),
        }}
      />

      {/* 2) History tab on the RIGHT */}
      <Tabs.Screen
        name="pastWorkouts"
        options={{
          tabBarIcon: ({ color }) => (
            <Image
              source={historyIcon}
              style={[styles.icon, { tintColor: color }]}
            />
          ),
        }}
      />

      {/* hidden but still routable */}
      <Tabs.Screen name="loading"  options={{ tabBarButton: () => null }} />
      <Tabs.Screen name="results"  options={{ tabBarButton: () => null }} />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  splashContainer: {
    flex: 1,
    backgroundColor: '#1C1C1C',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    width: SCREEN_WIDTH * 0.5,
    height: SCREEN_WIDTH * 0.5,
    tintColor: '#FF5A5A',
    marginBottom: 24,
  },
  splashText: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FF5A5A',
  },
  icon: {
    width: 28,
    height: 28,
    resizeMode: 'contain',
  },
});
