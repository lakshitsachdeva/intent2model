'use client';

import { useEffect, useState, useCallback } from 'react';

const TextType = ({
  texts = ['Welcome'],
  typingSpeed = 75,
  pauseDuration = 1500,
  deletingSpeed = 50,
  showCursor = true,
  cursorCharacter = '_',
  className = '',
}: {
  texts?: string[];
  typingSpeed?: number;
  pauseDuration?: number;
  deletingSpeed?: number;
  showCursor?: boolean;
  cursorCharacter?: string;
  className?: string;
}) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentCharIndex, setCurrentCharIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const [currentTextIndex, setCurrentTextIndex] = useState(0);
  const [cursorVisible, setCursorVisible] = useState(true);

  useEffect(() => {
    if (!showCursor) return;
    const interval = setInterval(() => {
      setCursorVisible((v) => !v);
    }, 500);
    return () => clearInterval(interval);
  }, [showCursor]);

  useEffect(() => {
    let timeout: NodeJS.Timeout;
    const currentText = texts[currentTextIndex];

    if (isDeleting) {
      if (displayedText === '') {
        setIsDeleting(false);
        setCurrentTextIndex((prev) => (prev + 1) % texts.length);
        setCurrentCharIndex(0);
        timeout = setTimeout(() => {}, pauseDuration);
      } else {
        timeout = setTimeout(() => {
          setDisplayedText((prev) => prev.slice(0, -1));
        }, deletingSpeed);
      }
    } else {
      if (currentCharIndex < currentText.length) {
        timeout = setTimeout(() => {
          setDisplayedText((prev) => prev + currentText[currentCharIndex]);
          setCurrentCharIndex((prev) => prev + 1);
        }, typingSpeed);
      } else {
        timeout = setTimeout(() => {
          setIsDeleting(true);
        }, pauseDuration);
      }
    }

    return () => clearTimeout(timeout);
  }, [currentCharIndex, displayedText, isDeleting, typingSpeed, deletingSpeed, pauseDuration, texts, currentTextIndex]);

  return (
    <span className={`inline-block ${className}`}>
      <span>{displayedText}</span>
      {showCursor && <span style={{ opacity: cursorVisible ? 1 : 0 }}>{cursorCharacter}</span>}
    </span>
  );
};

export default TextType;
