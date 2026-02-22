import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { AppModule } from './app.module';
import * as bodyParser from 'body-parser';
import { env } from '@matcha/env';

const JSON_BODY_LIMIT    = process.env.JSON_BODY_LIMIT    || '100mb';
const URLENCODED_LIMIT   = process.env.URLENCODED_BODY_LIMIT || '1mb';
const PORT               = parseInt(env.PORT, 10);
const REQUEST_TIMEOUT    = parseInt(process.env.REQUEST_TIMEOUT ?? '30000', 10);
const CORS_ORIGIN        = process.env.CORS_ORIGIN ? process.env.CORS_ORIGIN.split(',') : [];

async function bootstrap() {
  try {
    const app = await NestFactory.create(AppModule);

    // API versioning — all routes live under /api/v1
    app.setGlobalPrefix('api/v1');

    // CORS — dynamically allow custom arrays or any localhost/127.0.0.1 port.
    
    const allowedOrigins = CORS_ORIGIN.length > 0
      ? CORS_ORIGIN
      : [/^http:\/\/(localhost|127\.0\.0\.1):\d+$/];

    app.enableCors({
      origin: allowedOrigins,
      methods: ['GET', 'POST', 'DELETE', 'OPTIONS', 'PATCH'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Range'],
      credentials: true,
      exposedHeaders: ['Content-Range', 'X-Content-Duration'],
    });

    // Global validation pipe — rejects malformed payloads with HTTP 400
    app.useGlobalPipes(
      new ValidationPipe({
        whitelist: true,          // Strip unknown properties
        forbidNonWhitelisted: false, // Don't throw on extra fields from inference service
        transform: true,          // Auto-transform payloads to DTO types
        transformOptions: { enableImplicitConversion: true },
      }),
    );

    app.use(bodyParser.json({ limit: JSON_BODY_LIMIT }));
    app.use(bodyParser.urlencoded({ limit: URLENCODED_LIMIT, extended: true }));

    const server = await app.listen(PORT);
    server.setTimeout(REQUEST_TIMEOUT);

    console.log(`🚀 Orchestrator running on http://localhost:${PORT}/api/v1`);
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

bootstrap();
