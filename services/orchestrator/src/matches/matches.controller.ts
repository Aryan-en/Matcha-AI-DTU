import {
  Controller,
  Get,
  Post,
  Delete,
  Param,
  Body,
  UseInterceptors,
  UploadedFile,
  NotFoundException,
  BadRequestException,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Throttle } from '@nestjs/throttler';
import { diskStorage } from 'multer';
import { extname, join } from 'path';
import { MatchesService } from './matches.service';
import { Match } from '@prisma/client';
import 'multer';

@Controller('matches')
export class MatchesController {
  constructor(private readonly matchesService: MatchesService) {}

  // Stricter rate limit on upload — 5 uploads per minute to protect disk + inference queue
  @Throttle({ default: { ttl: 60_000, limit: 5 } })
  @Post('upload')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: join(process.cwd(), '..', '..', 'uploads'),
        filename: (req, file, cb) => {
          const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
          cb(null, `${uniqueSuffix}${extname(file.originalname)}`);
        },
      }),
      limits: {
        fileSize: 5 * 1024 * 1024 * 1024, // 5GB limit
      },
    }),
  )
  async uploadFile(@UploadedFile() file: Express.Multer.File): Promise<Match> {
    if (!file) throw new BadRequestException('No file provided');
    return this.matchesService.create(file);
  }

  @Get()
  async findAll(): Promise<Match[]> {
    return this.matchesService.findAll();
  }

  @Get(':id')
  async findOne(@Param('id') id: string): Promise<Match> {
    const match = await this.matchesService.findOne(id);
    if (!match) throw new NotFoundException(`Match ${id} not found`);
    return match;
  }

  @Post(':id/progress')
  async updateProgress(
    @Param('id') id: string,
    @Body() body: { progress: number },
  ) {
    if (typeof body.progress !== 'number') {
      throw new BadRequestException('progress must be a number');
    }
    return this.matchesService.updateProgress(id, body.progress);
  }

  @Post(':id/live-event')
  async addLiveEvent(@Param('id') id: string, @Body() body: object) {
    return this.matchesService.addLiveEvent(id, body);
  }

  @Post(':id/complete')
  async completeMatch(@Param('id') id: string, @Body() body: object) {
    if (!body) throw new BadRequestException('Payload required');
    return this.matchesService.completeMatch(id, body as any);
  }

  @Post(':id/reanalyze')
  async reanalyzeMatch(@Param('id') id: string): Promise<{ ok: boolean }> {
    const result = await this.matchesService.reanalyzeMatch(id);
    if (!result.ok) throw new NotFoundException(`Match ${id} not found`);
    return result;
  }

  @Delete(':id')
  async deleteMatch(@Param('id') id: string): Promise<{ ok: boolean }> {
    return this.matchesService.deleteMatch(id);
  }
}
