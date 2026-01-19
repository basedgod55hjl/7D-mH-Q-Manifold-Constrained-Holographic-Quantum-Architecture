# Manual Cleanup Required

The following directories were identified for removal but could not be deleted automatically due to file locks held by system processes (e.g., `compiler_service.exe`, `next-swc`):

- `projects/web_hologram/node_modules`
- `projects/web_hologram/.next`
- `target` (Rust build artifacts)

## Instructions

1. **Stop all running processes**: Ensure no development servers (Next.js, Rust runners) are active.
2. **Close IDE/Terminals**: Sometimes VS Code or terminals hold locks.
3. **Manually delete**:

   ```powershell
   Remove-Item -Recurse -Force projects/web_hologram/node_modules
   Remove-Item -Recurse -Force projects/web_hologram/.next
   Remove-Item -Recurse -Force target
   ```

4. **Verify**: Check that these folders are gone to reclaim disk space.
