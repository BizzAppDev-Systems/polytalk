/* SPDX-FileCopyrightText: 2026 BizzAppDev Systems Pvt. Ltd. */
/* SPDX-License-Identifier: AGPL-3.0-or-later */

(function () {
    const input = document.getElementById('text-source-input');
    if (!input) return;

    const sourceLanguage = document.getElementById('source-lang');
    const targetLanguage = document.getElementById('target-lang');
    const translateButton = document.getElementById('text-translate-btn');
    const copyButton = document.getElementById('text-copy-btn');
    const output = document.getElementById('text-translation-output');
    const status = document.getElementById('text-translation-status');
    const count = document.getElementById('text-character-count');
    const audioGrid = document.getElementById('text-audio-grid');
    const panel = document.getElementById('text-translation-card');
    const toggleButton = document.getElementById('text-mode-btn');
    const localize = (value) => window.PolyTalkI18n?.text(value) || value;
    const maxCharacters = Number(input.dataset.maxCharacters) || 5000;
    let sourceText = '';
    let translatedText = '';
    const audioPlayers = {
        source: {
            button: document.getElementById('text-source-speak-btn'),
            player: document.getElementById('text-source-audio-player'),
            panel: document.getElementById('text-source-audio-panel'),
            audioUrl: null,
            generating: false
        },
        translation: {
            button: document.getElementById('text-speak-btn'),
            player: document.getElementById('text-audio-player'),
            panel: document.getElementById('text-translation-audio-panel'),
            audioUrl: null,
            generating: false
        }
    };
    const requiredElements = {
        sourceLanguage,
        targetLanguage,
        translateButton,
        copyButton,
        output,
        status,
        count,
        audioGrid,
        panel,
        toggleButton
    };
    const missingRequiredElement = Object.entries(requiredElements).find(
        ([, element]) => !element
    );
    const incompleteAudioTrack = Object.entries(audioPlayers).find(
        ([, audio]) => !audio.button || !audio.player || !audio.panel
    );
    if (missingRequiredElement || incompleteAudioTrack) {
        const element = missingRequiredElement?.[0] || `audio track: ${incompleteAudioTrack[0]}`;
        console.warn(`[PolyTalk] Missing required DOM element: ${element}`);
        return;
    }

    function setTextModeOpen(open) {
        if (!open) clearAllAudio();
        panel.hidden = !open;
        toggleButton.setAttribute('aria-expanded', String(open));
        toggleButton.classList.toggle('active', open);
        document.body.classList.toggle('text-mode-active', open);
        if (open) {
            panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            window.setTimeout(() => input.focus(), 250);
        }
    }

    toggleButton.addEventListener('click', () => {
        if (panel.hidden) setTextModeOpen(true);
    });

    ['live-btn', 'conversation-btn', 'share-tab-btn'].forEach((id) => {
        document.getElementById(id)?.addEventListener(
            'click', () => setTextModeOpen(false)
        );
    });

    function setStatus(message, state = '') {
        status.textContent = localize(message);
        status.dataset.state = state;
    }

    function clearAudio(audio) {
        audio.player.pause();
        // Reset the media element after detaching its object URL-backed source.
        audio.player.removeAttribute('src');
        audio.player.load();
        audio.player.hidden = true;
        audio.panel.hidden = true;
        if (audio.audioUrl) URL.revokeObjectURL(audio.audioUrl);
        audio.audioUrl = null;
        audio.generating = false;
        syncAudioGridVisibility();
    }

    function clearAllAudio() {
        Object.values(audioPlayers).forEach(clearAudio);
    }

    function syncAudioGridVisibility() {
        audioGrid.hidden = !Object.values(audioPlayers).some(
            (audio) => !audio.panel.hidden
        );
    }

    function showOnlyAudioPanel(audio) {
        Object.values(audioPlayers).forEach((candidate) => {
            if (candidate !== audio) {
                candidate.player.pause();
                candidate.panel.hidden = true;
            }
        });
        audio.panel.hidden = !audio.audioUrl;
        syncAudioGridVisibility();
    }

    function invalidateOutput() {
        sourceText = '';
        translatedText = '';
        output.textContent = localize('Your translation will appear here.');
        copyButton.disabled = true;
        audioGrid.hidden = true;
        Object.values(audioPlayers).forEach((audio) => {
            audio.button.disabled = true;
            clearAudio(audio);
        });
    }

    const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
    function countGraphemes(str) {
        return [...segmenter.segment(str)].length;
    }
    function sliceGraphemes(str, max) {
        const segments = [...segmenter.segment(str)];
        return segments.slice(0, max).map(s => s.segment).join('');
    }

    input.addEventListener('input', () => {
        const characters = countGraphemes(input.value);
        if (characters > maxCharacters) input.value = sliceGraphemes(input.value, maxCharacters);
        count.textContent = `${Math.min(characters, maxCharacters)} / ${maxCharacters}`;
        translateButton.disabled = !input.value.trim();
        invalidateOutput();
    });

    function syncLanguages() {
        input.lang = sourceLanguage.value;
        output.lang = targetLanguage.value;
        invalidateOutput();
    }

    sourceLanguage.addEventListener('change', syncLanguages);
    targetLanguage.addEventListener('change', syncLanguages);
    document.getElementById('swap-langs-btn')?.addEventListener('click', () => {
        const previousSource = sourceLanguage.value;
        sourceLanguage.value = targetLanguage.value;
        targetLanguage.value = previousSource;
        syncLanguages();
    });

    translateButton.addEventListener('click', async () => {
        const currentSourceText = input.value.trim();
        if (!currentSourceText) return;
        invalidateOutput();
        input.disabled = true;
        translateButton.disabled = true;
        setStatus('Translating text...');
        try {
            const customInstruction = document.getElementById('custom-instruction-input')?.value.trim();
            const response = await fetch('/api/text/translations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: currentSourceText,
                    source_language: sourceLanguage.value,
                    target_language: targetLanguage.value,
                    custom_instruction: customInstruction || null
                })
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(payload.detail || 'Translation failed. Please try again.');
            sourceText = currentSourceText;
            translatedText = payload.translated_text;
            output.textContent = translatedText;
            output.lang = payload.target_language;
            copyButton.disabled = false;
            Object.values(audioPlayers).forEach((audio) => { audio.button.disabled = false; });
            setStatus('Translation ready. Choose an audio track to listen.', 'success');
        } catch (error) {
            output.textContent = localize('Your translation will appear here.');
            setStatus(error.message || 'Translation failed. Please try again.', 'error');
        } finally {
            input.disabled = false;
            translateButton.disabled = !input.value.trim();
            input.focus();
            const end = input.value.length;
            input.setSelectionRange(end, end);
        }
    });

    copyButton.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(translatedText);
            setStatus('Translation copied.', 'success');
        } catch (_error) {
            setStatus('Could not copy the translation.', 'error');
        }
    });

    async function prepareAudio(kind) {
        const audio = audioPlayers[kind];
        const textToSpeak = kind === 'source' ? sourceText : translatedText;
        const language = kind === 'source' ? sourceLanguage.value : targetLanguage.value;
        if (!textToSpeak || audio.generating) return false;
        audio.generating = true;
        Object.values(audioPlayers).forEach((candidate) => {
            candidate.button.disabled = true;
        });
        setStatus(kind === 'source' ? 'Generating source audio...' : 'Generating translated audio...');
        try {
            const response = await fetch('/api/text/speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textToSpeak, language })
            });
            if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                throw new Error(payload.detail || 'Audio generation failed. Please try again.');
            }
            clearAudio(audio);
            audio.audioUrl = URL.createObjectURL(await response.blob());
            audio.player.src = audio.audioUrl;
            audio.player.hidden = false;
            audio.panel.hidden = false;
            syncAudioGridVisibility();
            setStatus(kind === 'source' ? 'Source audio ready.' : 'Translated audio ready.', 'success');
            return true;
        } catch (error) {
            setStatus(error.message || 'Audio generation failed. Please try again.', 'error');
            return false;
        } finally {
            audio.generating = false;
            Object.values(audioPlayers).forEach((candidate) => {
                candidate.button.disabled = false;
            });
        }
    }

    async function speak(kind) {
        const audio = audioPlayers[kind];
        if (Object.values(audioPlayers).some((candidate) => candidate.generating)) return;
        showOnlyAudioPanel(audio);
        if (!audio.audioUrl && !(await prepareAudio(kind))) return;
        await audio.player.play().catch((error) => {
            console.warn(
                '[PolyTalk] Audio playback failed:',
                error?.name || 'UnknownError',
                error?.message || 'No error details available'
            );
            setStatus('Audio playback was blocked by the browser.', 'error');
        });
    }

    Object.entries(audioPlayers).forEach(([kind, audio]) => {
        audio.button.addEventListener('click', () => speak(kind));
        audio.player.addEventListener('play', () => showOnlyAudioPanel(audio));
    });
    window.addEventListener('beforeunload', clearAllAudio);
    syncLanguages();
    window._PolyTalkTextTranslation = { clearAudio: clearAllAudio };
}());
