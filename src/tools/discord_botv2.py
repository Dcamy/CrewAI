# iChain/src/iChain/tools/discord_bot_v2.py

import discord
from discord.ext import commands
import os
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscordBotTool:
    def __init__(self, token):
        """
        Initializes the Discord bot with necessary intents and configurations.
        """
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.token = token

        # Register event handlers and commands
        self.bot.event(self.on_ready)
        self.bot.command(
            name="info", help="Provides information based on the user's query"
        )(self.info)

    async def on_ready(self):
        """
        Event handler for when the bot logs in successfully.
        """
        logger.info(f"Logged in as {self.bot.user}!")

    async def info(self, ctx, *, query: str):
        """
        A slash command to process queries using CrewAI's system, returning insights directly.
        """
        # Example interaction, replace with actual CrewAI integration
        response = f"Received your query: {query}"
        await ctx.send(response)

    def run(self):
        """
        Runs the Discord bot.
        """
        self.bot.run(self.token)