const path = require('path');

/** @type {import('next').NextConfig} */
const nextConfig = {
    transpilePackages: [
        '@matcha/shared',
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
