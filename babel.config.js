module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      ['react-native-worklets-core/plugin'],
      [
        'react-native-reanimated/plugin', // add this line
      ],
      ['module:react-native-dotenv'],
    ],
  };
};
