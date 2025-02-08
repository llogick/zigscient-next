Using the Zig Compiler's Incremental Semantic Analysis as a Foundation for Near-instant Code Feedback via LSP

![zigscient-next](https://github.com/user-attachments/assets/a170b5f0-5aaf-47f3-8935-a27ef3684f52)

**Status**
----------------

This project is a proof-of-concept/viability effort. The intention is to demonstrate the potential of using Zig's incremental semantic analysis for near-instant code feedback via LSP, and as a potential bridge between today and when the Zig Compile Server emerges.

**Important Notes**
-------------------

### Server Requirements

The server requires a valid workspace folder path to be passed in the `initialize` request by the editor. If an editor does not provide this information, the server will fall back to basic functionality.

### Correct Modules Lookup

To ensure correct modules lookup, please refer to the [wiki page](https://github.com/llogick/zigscient/wiki/Modules:-Switching-%60root_id%60).

## Building
>[!NOTE]
> This is a resource-intensive piece of software, so a capable CPU with good single-thread performance is recommended.

```bash
zig build -Doptimize=ReleaseFast --zig-lib-dir ./lib/
```
