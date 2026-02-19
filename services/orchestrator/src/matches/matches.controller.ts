import { Controller, Get, Post, Param, UseInterceptors, UploadedFile, Body } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { MatchesService } from './matches.service';
import { Match } from '@prisma/client';
import 'multer';

@Controller('matches')
export class MatchesController {
  constructor(private readonly matchesService: MatchesService) {}

  @Post('upload')
  @UseInterceptors(FileInterceptor('file'))
  async uploadFile(@UploadedFile() file: any): Promise<Match> {
    return this.matchesService.create(file);
  }

  @Get()
  async findAll(): Promise<Match[]> {
    return this.matchesService.findAll();
  }

  @Get(':id')
  async findOne(@Param('id') id: string): Promise<Match | null> {
    return this.matchesService.findOne(id);
  }

  @Post(':id/progress')
  async updateProgress(@Param('id') id: string, @Body() body: { progress: number }) {
    return this.matchesService.updateProgress(id, body.progress);
  }
}
