# Recording the README Demo GIF

Goal: a 10–15 second loop of `python -m src.demo_cli` streaming the orbit pass.
Save the result as `docs/assets/demo.gif` — the README references that path.

## On Windows (recommended: ScreenToGif)

1. Install **[ScreenToGif](https://www.screentogif.com/)** — free, ~3 MB.
2. Open a clean PowerShell window. Maximize, set font to **Cascadia Mono 14pt** for legibility.
3. Run the demo first to warm the import cache:
   ```powershell
   python -m src.demo_cli --slow 0
   ```
4. Now record the real take:
   - Open ScreenToGif → **Recorder**.
   - Frame the terminal so the cyan banner and final summary panel both fit.
   - Set FPS to **15** (smaller file, plenty smooth for terminal output).
   - Hit record, run:
     ```powershell
     python -m src.demo_cli --slow 0.35
     ```
   - Stop recording right after the green "ORBIT PASS COMPLETE" panel renders.
5. In ScreenToGif's editor: **Edit → Reduce Frame Count** → keep every 2nd frame. **File → Save as GIF** → enable **Loop**.
6. Target output size: under **2 MB**. If larger, lower FPS to 10 or crop tighter.
7. Save as `docs/assets/demo.gif`.

## Alternative: VHS (scriptable, deterministic)

If you want the GIF reproducible from a script, use [charm.sh/vhs](https://github.com/charmbracelet/vhs):

```tape
# demo.tape
Output docs/assets/demo.gif
Set FontSize 14
Set Width 1100
Set Height 720
Set Theme "Dracula"
Type "python -m src.demo_cli --slow 0.35"
Enter
Sleep 6s
```

```powershell
vhs demo.tape
```

## Smoke-test the GIF

After saving:
- Open `README.md` in VS Code preview — the GIF should auto-play in the hero block.
- Push to GitHub and confirm it renders on the public page (sometimes the first push takes ~30s for the CDN to pick it up).
