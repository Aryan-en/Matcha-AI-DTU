import { Injectable, Logger } from '@nestjs/common';
import { PrismaClient, Match } from '@prisma/client';
import { HttpService } from '@nestjs/axios';
import { EventsGateway } from '../events/events.gateway';
import { firstValueFrom } from 'rxjs';
import * as fs from 'fs';
import * as path from 'path';
import 'multer';

@Injectable()
export class MatchesService {
  private prisma: PrismaClient;
  private readonly logger = new Logger(MatchesService.name);

  constructor(
    private eventsGateway: EventsGateway,
    private httpService: HttpService,
  ) {
    this.prisma = new PrismaClient();
  }

  async create(file: any): Promise<Match> {
    const fileName = `${Date.now()}-${file.originalname}`;
    const uploadsDir = path.join(process.cwd(), '..', '..', 'uploads');
    
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }

    const filePath = path.join(uploadsDir, fileName);
    fs.writeFileSync(filePath, file.buffer);

    // This is the path as seen by the Inference service inside Docker
    const containerVideoPath = `/app/uploads/${fileName}`;
    const publicUrl = `http://localhost:4000/uploads/${fileName}`; // Mock public URL
    
    const match = await this.prisma.match.create({
      data: {
        uploadUrl: publicUrl,
        status: 'UPLOADED',
        duration: 0, 
      },
    });

    this.triggerInference(match.id, containerVideoPath);

    return match;
  }

  async triggerInference(matchId: string, videoUrl: string) {
    const inferenceUrl = process.env.INFERENCE_URL || 'http://localhost:8000';
    try {
      this.logger.log(`Triggering inference for match ${matchId} at ${videoUrl}`);
      await firstValueFrom(
        this.httpService.post(`${inferenceUrl}/api/v1/analyze`, {
          match_id: matchId,
          video_url: videoUrl, // This needs to be a real path/URL accessible to Docker
        }) as any,
      );
    } catch (error) {
      this.logger.error(`Failed to trigger inference: ${error.message}`);
    }
  }

  async findAll(): Promise<Match[]> {
    return this.prisma.match.findMany({
      orderBy: { createdAt: 'desc' },
    });
  }

  async findOne(id: string): Promise<Match | null> {
    return this.prisma.match.findUnique({
      where: { id },
      include: { events: true },
    });
  }

  async updateProgress(id: string, progress: number) {
    this.eventsGateway.server.to(id).emit('progress', { progress });
  }
}
