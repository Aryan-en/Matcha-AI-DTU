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
  UseGuards,
  Req,
} from '@nestjs/common';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { OptionalJwtAuthGuard } from '../auth/optional-jwt-auth.guard';
import { FileInterceptor } from '@nestjs/platform-express';
import { Throttle } from '@nestjs/throttler';
import { diskStorage } from 'multer';
import { extname, join } from 'path';
import { MatchesService } from './matches.service';
import { Match } from "@matcha/database";
import 'multer';

@Controller('matches')
export class MatchesController {
  constructor(private readonly matchesService: MatchesService) {}

  // Stricter rate limit on upload — 5 uploads per minute to protect disk + inference queue
  @UseGuards(JwtAuthGuard)
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
  async uploadFile(@UploadedFile() file: Express.Multer.File, @Req() req: any): Promise<Match> {
    if (!file) throw new BadRequestException('No file provided');
    return this.matchesService.create(file, req.user.userId);
  }

  @UseGuards(JwtAuthGuard)
  @Throttle({ default: { ttl: 60_000, limit: 10 } })
  @Post('youtube')
  async uploadYoutube(@Body() body: { url: string }, @Req() req: any): Promise<Match> {
    if (!body || !body.url) {
      throw new BadRequestException('YouTube URL is required');
    }
    try {
      // Basic validation for youtube/youtu.be domains
      const url = new URL(body.url);
      if (!url.hostname.includes('youtube.com') && !url.hostname.includes('youtu.be')) {
        throw new BadRequestException('Invalid YouTube URL');
      }
    } catch {
      throw new BadRequestException('Invalid URL format');
    }
    return this.matchesService.createFromYoutube(body.url, req.user.userId);
  }

  @UseGuards(OptionalJwtAuthGuard)
  @Get()
  async findAll(@Req() req: any): Promise<Match[]> {
    return this.matchesService.findAll(req.user?.userId);
  }

  @UseGuards(OptionalJwtAuthGuard)
  @Get(':id')
  async findOne(@Param('id') id: string, @Req() req: any): Promise<Match> {
    const match = await this.matchesService.findOne(id, req.user?.userId);
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

  @UseGuards(JwtAuthGuard)
  @Post(':id/reanalyze')
  async reanalyzeMatch(@Param('id') id: string, @Req() req: any): Promise<{ ok: boolean }> {
    const result = await this.matchesService.reanalyzeMatch(id, req.user.userId);
    if (!result.ok) throw new NotFoundException(`Match ${id} not found`);
    return result;
  }

  @UseGuards(JwtAuthGuard)
  @Delete(':id')
  async deleteMatch(@Param('id') id: string, @Req() req: any): Promise<{ ok: boolean }> {
    return this.matchesService.deleteMatch(id, req.user.userId);
  }
}
