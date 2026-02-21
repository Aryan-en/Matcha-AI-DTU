import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import * as bodyParser from 'body-parser';

const JSON_BODY_LIMIT = process.env.JSON_BODY_LIMIT || '100mb';
const URLENCODED_BODY_LIMIT = process.env.URLENCODED_BODY_LIMIT || '1mb';
const PORT = parseInt(process.env.PORT ?? '4000', 10);
const REQUEST_TIMEOUT = parseInt(process.env.REQUEST_TIMEOUT ?? '30000', 10);

async function bootstrap() {
  try {
    const app = await NestFactory.create(AppModule);
    app.enableCors();
    
    app.use(bodyParser.json({ limit: JSON_BODY_LIMIT }));
    app.use(bodyParser.urlencoded({ limit: URLENCODED_BODY_LIMIT, extended: true }));
    
    const server = await app.listen(PORT);
    server.setTimeout(REQUEST_TIMEOUT);
    
    console.log(`🚀 NestJS server running on port ${PORT}`);
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

bootstrap();
