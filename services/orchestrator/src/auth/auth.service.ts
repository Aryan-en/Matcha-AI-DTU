import { Injectable, UnauthorizedException, ConflictException } from '@nestjs/common';
import { PrismaClient } from "@matcha/database";
import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcryptjs';

@Injectable()
export class AuthService {
  private prisma: PrismaClient;

  constructor(
    private jwtService: JwtService,
  ) {
    this.prisma = new PrismaClient();
    console.log(`[AUTH] JWT_SECRET configured: ${!!process.env.JWT_SECRET}`);
  }

  async validateUser(email: string, pass: string): Promise<any> {
    const lowerEmail = email.toLowerCase();
    console.log(`[AUTH] Attempting validation for: ${lowerEmail}`);
    const user = await this.prisma.user.findUnique({ where: { email: lowerEmail } });
    if (!user) {
      console.log(`[AUTH] User not found: ${email}`);
      return null;
    }
    console.log(`[AUTH] User found. Hash length: ${user.password.length}`);
    const isMatch = await bcrypt.compare(pass, user.password);
    console.log(`[AUTH] Bcrypt comparison result: ${isMatch}`);
    
    if (isMatch) {
      console.log(`[AUTH] Success for: ${email}`);
      const { password, ...result } = user;
      return result;
    }
    console.log(`[AUTH] Failure for: ${email}`);
    return null;
  }

  async getUserById(id: string): Promise<any> {
    const user = await this.prisma.user.findUnique({ where: { id } });
    if (!user) return null;
    const { password, ...result } = user;
    return result;
  }

  async login(user: any) {
    const payload = { email: user.email, sub: user.id };
    return {
      access_token: this.jwtService.sign(payload),
      user: {
        id: user.id,
        email: user.email,
        name: user.firstName && user.lastName 
          ? `${user.firstName} ${user.lastName}` 
          : user.name || user.email.split('@')[0],
      }
    };
  }

  async register(data: any): Promise<any> {
    const lowerEmail = data.email.toLowerCase();
    const existing = await this.prisma.user.findUnique({
      where: { email: lowerEmail },
    });
    
    if (existing) {
      throw new ConflictException('Email already in use');
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(data.password, salt);
    console.log(`[AUTH] Registering user: ${lowerEmail}. Salt: ${salt}. Hash length: ${hashedPassword.length}`);

    const user = await this.prisma.user.create({
      data: {
        email: lowerEmail,
        firstName: data.firstName,
        lastName: data.lastName,
        password: hashedPassword,
      },
    });

    const { password, ...result } = user;
    return result;
  }
}
