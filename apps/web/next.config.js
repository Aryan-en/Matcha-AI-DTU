/** @type {import('next').NextConfig} */
const nextConfig = {
    transpilePackages: ['socket.io-client', 'engine.io-client', 'engine.io-parser'],
};

module.exports = nextConfig;
