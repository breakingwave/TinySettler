# Linux shell and editor

The Mac zsh and LazyVim configurations were ported to the VPS `rexdev` account
and the coding container's persistent `/root` home. The Mac copies, shell
history, credentials, and SSH keys were not changed or copied as part of this port.

New SSH shells and new `~/.local/bin/coding` shells use zsh. In an existing
idle shell, switch with `exec zsh -l`. Inside the coding container:

```sh
cd /workspace/TinySettler
n .
```

Your prompt, aliases, history settings, autosuggestions, syntax highlighting,
zoxide, Monokai theme, Telescope, Oil, and custom key mappings are retained.
In particular, `j/k/l/;` mean left/down/up/right, `jj` exits insert mode,
and Space-w saves. `y` launches Yazi and follows its selected directory.

Linux adaptations: GNU `ls --color=auto`, Linux plugin paths, home-relative
tool paths, and a `[coding]` prompt inside Docker. The headless Linux editor
uses the dark theme instead of polling a desktop appearance service. Yanking
uses OSC 52 through Rex; paste Mac clipboard text with Rex's normal paste
shortcut. Editor `p` uses its internal register. End-to-end clipboard behavior
depends on the terminal and was not verified by the headless checks.

Neovim 0.12.4, Tree-sitter CLI 0.26.11, BasedPyright 1.39.10, Yazi 26.5.6,
Glow 2.1.2, and Lazygit 0.64.1 match the inspected Mac versions. Linux shell
dependencies come from Ubuntu/Debian packages. Neovim plugins were restored
from the Mac `lazy-lock.json`; desktop appearance polling is disabled on Linux.
The container needs a locally compiled Tree-sitter CLI because the vendor's
prebuilt executable requires a newer glibc than Debian Bookworm provides.

Configurations live in `~/.zshrc` and `~/.config/nvim`. Tools and plugins live
under `~/.local`; the container's home volume retains them across recreation.
The host and container each keep their own editor state. These dotfiles are a
one-time port, not an automatic Mac-dotfiles synchronization rule.

Backups are in `~/.local/state/rex-editor/before-*`; the host's original coding
launcher is `~/.local/state/rex-editor/coding.before-zsh`. Installation logs,
verification output, and release hashes are under `~/.local/state/rex-editor`
and `~/.local/share/rex-editor`. The selected-dotfile archive and deployment
scripts are under `~/.local/share/rex-coding` on the host. Container system
packages are recorded in that directory's Dockerfile as well.

Both environments passed zsh startup/plugin checks and headless Neovim tests
for the custom mappings, Monokai theme, Telescope native search, Oil, Python
syntax parsing, and BasedPyright attachment. The container installed all 31
configured syntax parsers. Codex's sandbox command test also still passes.
No terminal clipboard round trip was tested.

The Compose recipe uses image `rex-coding:editor-20261110` for future
recreation, retaining the repaired seccomp policy and existing home/workspace
mounts. `compose.before-editor.json` preserves the previous recipe. The current
container and Rex/Docker services were not restarted for this port.

Mac-specific Homebrew setup and Antigravity paths were not installed on Linux.
The existing optional `termpdf` wrapper remains in zsh, but TermPDF itself was
not installed. The optional `wt` initialization only runs if that tool exists.

References: [LazyVim requirements](https://www.lazyvim.org/),
[Neovim installation](https://github.com/neovim/neovim/blob/master/INSTALL.md).
