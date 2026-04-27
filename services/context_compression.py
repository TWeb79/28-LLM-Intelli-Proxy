"""
Context Compression Engine for LLM IntelliProxy.

Provides intelligent prompt compression to reduce token usage while preserving
critical information. Supports three compression levels and mode-specific rules.
"""
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class CompressionSettings:
    """Configuration for context compression."""
    
    def __init__(
        self,
        enabled: bool = True,
        level: str = "medium",  # low, medium, high
        max_tokens: int = 4096,
        mode: str = "general"  # coding, chat, general
    ):
        self.enabled = enabled
        self.level = level.lower()
        self.max_tokens = max_tokens
        self.mode = mode.lower()


class ContextCompressionEngine:
    """Intelligent context compression with mode-specific rules."""
    
    # ANSI escape code patterns
    ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*[mK]')
    
    # Critical patterns to never remove
    CRITICAL_PATTERNS = [
        r'error[:\s]', r'exception[:\s]', r'fail(?:ure|ed)?[:\s]',
        r'warning[:\s]', r'traceback', r'stack trace',
        r'file[:\s]', r'line[:\s]', r'path[:\s]',
        r'return[:\s]', r'exit[:\s]', r'crash[:\s]'
    ]
    
    # Noise patterns to always remove
    NOISE_PATTERNS = [
        r'\[\s*\d+%\s*\]',  # Progress bars
        r'\.{3,}',  # Ellipsis sequences
        r'={3,}',  # Separator lines
        r'-{3,}',  # Dash separators
        r'\*{3,}',  # Asterisk separators
    ]
    
    def __init__(self):
        self.critical_regex = [re.compile(p, re.IGNORECASE) for p in self.CRITICAL_PATTERNS]
        self.noise_regex = [re.compile(p) for p in self.NOISE_PATTERNS]
    
    def compress_context(
        self, 
        messages: List[Dict[str, str]], 
        settings: CompressionSettings
    ) -> List[Dict[str, str]]:
        """
        Compress context according to settings.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            settings: CompressionSettings instance
            
        Returns:
            Compressed messages with prepended context sections
        """
        if not settings.enabled:
            return self._basic_cleanup(messages)
        
        # Extract and compress content
        original_content = self._extract_content(messages)
        compressed_content = self._apply_compression(original_content, settings)
        
        # Build result with context sections
        result = []
        
        # Add goal section
        goal = self._extract_goal(original_content)
        if goal:
            result.append({"role": "system", "content": f"[GOAL] {goal}"})
        
        # Add key context section
        key_context = self._extract_key_context(original_content, settings)
        if key_context:
            result.append({"role": "system", "content": f"[KEY CONTEXT] {key_context}"})
        
        # Add compressed input
        result.append({"role": "system", "content": f"[COMPRESSED INPUT] {compressed_content}"})
        
        # Add the last user message to maintain conversation flow
        if messages and messages[-1].get("role") == "user":
            result.append(messages[-1])
        
        # Enforce token limit
        result = self._enforce_token_limit(result, settings.max_tokens)
        
        return result
    
    def _basic_cleanup(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Basic cleanup without compression."""
        cleaned = []
        seen_content = set()
        
        for msg in messages:
            if not msg or "content" not in msg:
                continue
                
            content = msg["content"]
            
            # Remove ANSI codes
            content = self.ANSI_PATTERN.sub("", content)
            
            # Remove noise patterns
            for pattern in self.noise_regex:
                content = pattern.sub("", content)
            
            # Remove exact duplicates
            if content in seen_content:
                continue
            seen_content.add(content)
            
            # Skip empty messages
            if content.strip():
                cleaned.append({"role": msg.get("role", "user"), "content": content})
        
        return cleaned
    
    def _extract_content(self, messages: List[Dict[str, str]]) -> str:
        """Extract all content into a single string."""
        return "\n".join([msg.get("content", "") for msg in messages if msg.get("content")])
    
    def _apply_compression(self, content: str, settings: CompressionSettings) -> str:
        """Apply compression based on level and mode."""
        if not content.strip():
            return content
        
        # Remove noise first
        content = self._remove_noise(content)
        
        if settings.level == "low":
            return self._low_compression(content)
        elif settings.level == "medium":
            return self._medium_compression(content, settings.mode)
        elif settings.level == "high":
            return self._high_compression(content, settings.mode)
        else:
            return content
    
    def _remove_noise(self, content: str) -> str:
        """Remove noise patterns while preserving critical content."""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip noise patterns
            is_noise = any(pattern.search(line) for pattern in self.noise_regex)
            if is_noise and not self._is_critical_line(line):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_critical_line(self, line: str) -> bool:
        """Check if line contains critical information."""
        for pattern in self.critical_regex:
            if pattern.search(line):
                return True
        return False
    
    def _low_compression(self, content: str) -> str:
        """Low compression: remove noise and duplicates only."""
        lines = content.split('\n')
        seen = set()
        result = []
        
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                result.append(line)
        
        return '\n'.join(result)
    
    def _medium_compression(self, content: str, mode: str) -> str:
        """Medium compression: summarize verbose content."""
        lines = content.split('\n')
        result = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if self._is_critical_line(line):
                result.append(line)
            elif len(line) > 200:
                # Summarize long lines
                summary = self._summarize_line(line, mode)
                if summary != line:
                    result.append(f"[SUMMARY] {summary}")
                else:
                    result.append(line)
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _high_compression(self, content: str, mode: str) -> str:
        """High compression: retain only critical information."""
        lines = content.split('\n')
        critical_lines = []
        
        for line in lines:
            line = line.strip()
            if line and self._is_critical_line(line):
                critical_lines.append(line)
        
        if not critical_lines:
            # If no critical lines, create dense summary
            return self._create_dense_summary(content, mode)
        
        return '\n'.join(critical_lines)
    
    def _summarize_line(self, line: str, mode: str) -> str:
        """Summarize a single line based on mode."""
        if mode == "coding":
            return self._summarize_coding_line(line)
        elif mode == "chat":
            return self._summarize_chat_line(line)
        else:
            return self._summarize_general_line(line)
    
    def _summarize_coding_line(self, line: str) -> str:
        """Summarize coding-related content."""
        # Extract key info from git output, test results, etc.
        if "test" in line.lower():
            if "fail" in line.lower():
                return "Tests failed"
            elif "pass" in line.lower():
                return "Tests passed"
        
        if "git" in line.lower():
            if "changed" in line.lower():
                return "Files changed"
        
        return line[:100] + "..." if len(line) > 100 else line
    
    def _summarize_chat_line(self, line: str) -> str:
        """Summarize chat-related content."""
        # Remove pleasantries and focus on key facts
        pleasantries = ["please", "thank you", "hello", "hi", "hey"]
        words = line.lower().split()
        
        if any(p in words for p in pleasantries):
            # Extract the core request
            core = re.sub(r'\b(?:please|thank you|hello|hi|hey)\b', '', line, flags=re.IGNORECASE)
            return core.strip()
        
        return line[:80] + "..." if len(line) > 80 else line
    
    def _summarize_general_line(self, line: str) -> str:
        """Summarize general content."""
        # One sentence per concept
        sentences = line.split('.')
        if len(sentences) > 1:
            return sentences[0].strip() + "."
        return line[:60] + "..." if len(line) > 60 else line
    
    def _create_dense_summary(self, content: str, mode: str) -> str:
        """Create a dense summary when no critical lines found."""
        words = content.split()
        if len(words) <= 10:
            return content
        
        # Extract key concepts
        key_words = [w for w in words if len(w) > 3 and w.isalnum()]
        unique_words = list(dict.fromkeys(key_words))  # Preserve order
        
        return f"Content: {' '.join(unique_words[:15])}..."
    
    def _extract_goal(self, content: str) -> str:
        """Extract the main goal from content."""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('>', '-', '#')):
                # Look for task/request indicators
                if any(indicator in line.lower() for indicator in ['write', 'create', 'explain', 'analyze', 'debug']):
                    return line[:100]  # Truncate if too long
        
        # Fallback to first meaningful line
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                return line[:100]
        
        return "Process request"
    
    def _extract_key_context(self, content: str, settings: CompressionSettings) -> str:
        """Extract key context from content."""
        lines = content.split('\n')
        context_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for constraints, requirements, or important state
            if any(keyword in line.lower() for keyword in ['constraint', 'requirement', 'must', 'should', 'error', 'warning']):
                context_lines.append(line)
            elif self._is_critical_line(line):
                context_lines.append(line)
        
        if not context_lines:
            return ""
        
        return "; ".join(context_lines)[:200]  # Limit length
    
    def _enforce_token_limit(self, messages: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
        """Enforce token limit, preserving critical content."""
        # Simple token estimation: ~4 chars per token
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        estimated_tokens = total_chars // 4
        
        if estimated_tokens <= max_tokens:
            return messages
        
        # Need to truncate - start with critical content
        result = []
        current_chars = 0
        max_chars = max_tokens * 4
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Always include critical sections
            if content.startswith(('[GOAL]', '[KEY CONTEXT]')):
                result.append(msg)
                current_chars += len(content)
                continue
            
            # For other content, truncate if needed
            remaining = max_chars - current_chars
            if remaining <= 50:  # Leave room for truncation marker
                break
            
            if len(content) > remaining:
                truncated = content[:remaining-50] + " [TRUNCATED: {} low-priority tokens removed]".format(
                    (total_chars - max_chars) // 4
                )
                result.append({"role": msg.get("role"), "content": truncated})
                break
            else:
                result.append(msg)
                current_chars += len(content)
        
        return result


# Global instance
_compression_engine = None


def get_compression_engine() -> ContextCompressionEngine:
    """Get the global compression engine instance."""
    global _compression_engine
    if _compression_engine is None:
        _compression_engine = ContextCompressionEngine()
    return _compression_engine


def compress_context(messages: List[Dict[str, str]], settings: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convenience function for compression.
    
    Args:
        messages: List of message dicts
        settings: Dict with compression settings
        
    Returns:
        Compressed messages
    """
    engine = get_compression_engine()
    compression_settings = CompressionSettings(
        enabled=settings.get("enabled", True),
        level=settings.get("level", "medium"),
        max_tokens=settings.get("max_tokens", 4096),
        mode=settings.get("mode", "general")
    )
    return engine.compress_context(messages, compression_settings)