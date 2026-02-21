const path = require('path');

/** @type {import('next').NextConfig} */
const nextConfig = {
    transpilePackages: [
        'socket.io-client', 
        'engine.io-client', 
        'engine.io-parser',
        '@tensorflow-models/coco-ssd',
        '@tensorflow/tfjs',
        '@tensorflow/tfjs-core',
        '@tensorflow/tfjs-converter',
        '@tensorflow/tfjs-backend-webgl',
        '@tensorflow/tfjs-backend-cpu'
    ],
    webpack: (config) => {
        // In a monorepo, hoisted packages live in the workspace root node_modules.
        config.resolve.alias = {
            ...config.resolve.alias,
            '@tensorflow/tfjs-core': path.resolve(__dirname, '../../node_modules/@tensorflow/tfjs-core'),
            '@tensorflow/tfjs-converter': path.resolve(__dirname, '../../node_modules/@tensorflow/tfjs-converter'),
            '@tensorflow/tfjs-backend-webgl': path.resolve(__dirname, '../../node_modules/@tensorflow/tfjs-backend-webgl'),
            '@tensorflow/tfjs-backend-cpu': path.resolve(__dirname, '../../node_modules/@tensorflow/tfjs-backend-cpu'),
        };
        return config;
    },
    async headers() {
        return [
            {
                source: '/(.*)',
                headers: [
                    {
                        key: 'Cross-Origin-Opener-Policy',
                        value: 'same-origin',
                    },
                    {
                        key: 'Cross-Origin-Embedder-Policy',
                        value: 'require-corp',
                    },
                ],
            },
        ];
    },
};

module.exports = nextConfig;
