// SplashScreen.tsx
import React, { useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  Dimensions,
} from 'react-native';
import type { StackNavigationProp } from '@react-navigation/stack';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type RootStackParamList = {
  Splash: undefined;
  Upload: undefined;
  // ...other routes
};

type SplashScreenProps = {
  navigation: StackNavigationProp<RootStackParamList, 'Splash'>;
};

const SplashScreen: React.FC<SplashScreenProps> = ({ navigation }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.replace('Upload');
    }, 3000); // 3 seconds

    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <Image
        source={require('../../assets/logo.png')} // path to your logo PNG
        style={styles.logo}
        resizeMode="contain"
      />
      <Text style={styles.text}>Welcome to Align</Text>
    </View>
  );
};

export default SplashScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1C1C1C',    // dark background
    alignItems: 'center',
    justifyContent: 'center',
  },
  logo: {
    width: SCREEN_WIDTH * 0.5,      // 50% of screen width
    height: SCREEN_WIDTH * 0.5,     // keep square
    marginBottom: 24,
    tintColor: '#FF5A5A',           // coral accent
  },
  text: {
    fontSize: 28,
    fontWeight: '700',
    color: '#FF5A5A',               // same coral for text
  },
});
