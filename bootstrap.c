#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

static const char *get_c_compiler(void) {
    const char *cc = getenv("CC");
    return (cc == NULL) ? "cc" : cc;
}

static void panic(const char *reason) {
    fprintf(stderr, "%s\n", reason);
    abort();
}

#if defined(__WIN32__)
#error TODO write the functionality for executing child process into this build script
#else

#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>

static void run(char **argv) {
    pid_t pid = fork();
    if (pid == -1)
        panic("fork failed");
    if (pid == 0) {
        // child
        execvp(argv[0], argv);
        exit(1);
    }

    // parent

    int status;
    waitpid(pid, &status, 0);

    if (!WIFEXITED(status))
        panic("child process crashed");

    if (WEXITSTATUS(status) != 0)
        panic("child process failed");
}
#endif

static void print_and_run(const char **argv) {
    fprintf(stderr, "%s", argv[0]);
    for (const char **arg = argv + 1; *arg; arg += 1) {
        fprintf(stderr, " %s", *arg);
    }
    fprintf(stderr, "\n");
    run((char **)argv);
}

static const char *get_host_os(void) {
    const char *host_os = getenv("ZIG_HOST_TARGET_OS");
    if (host_os != NULL) return host_os;
#if defined(__WIN32__)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#elif defined(__FreeBSD__)
    return "freebsd";
#elif defined(__DragonFly__)
    return "dragonfly";
#elif defined(__HAIKU__)
    return "haiku";
#else
    panic("unknown host os, specify with ZIG_HOST_TARGET_OS");
#endif
}

static const char *get_host_arch(void) {
    const char *host_arch = getenv("ZIG_HOST_TARGET_ARCH");
    if (host_arch != NULL) return host_arch;
#if defined(__x86_64__ )
    return "x86_64";
#elif defined(__aarch64__)
    return "aarch64";
#else
    panic("unknown host arch, specify with ZIG_HOST_TARGET_ARCH");
#endif
}

static const char *get_host_abi(void) {
    const char *host_abi = getenv("ZIG_HOST_TARGET_ABI");
    return (host_abi == NULL) ? "" : host_abi;
}

static const char *get_host_triple(void) {
    const char *host_triple = getenv("ZIG_HOST_TARGET_TRIPLE");
    if (host_triple != NULL) return host_triple;
    static char global_buffer[100];
    sprintf(global_buffer, "%s-%s%s", get_host_arch(), get_host_os(), get_host_abi());
    return global_buffer;
}

int main(int argc, char **argv) {
    const char *cc = get_c_compiler();
    const char *host_triple = get_host_triple();

    // GCC versions 10.0--15.1 have a miscompilation where some bytes of a union may get clobbered
    // depending on the union layout and the order in which types are defined. This miscompilation
    // affects the output of the C backend, and thus can affect the bootstrap process. Specifically,
    // we observe that using the self-hosted x86_64 backend in 'zig2' will cause all function calls
    // to be relocated incorrectly, causing immediate crashes on any binary produced by it.
    //
    // The only reliable workaround for this bug is to disable the optimization pass containing it,
    // so here we check for a CLI flag requesting that workaround.
    //
    // The upstream bug is fixed in GCC version 15.2 onwards (and was also backported to the 13 and
    // 14 branches). Once this bug is no longer widespread, we can remove this CLI flag.
    //
    // Upstream bug report: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=119085
    bool workaround_gcc_sra_miscomp = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--workaround-gcc-sra-miscomp")) {
            workaround_gcc_sra_miscomp = true;
        }
    }

    {
        const char *child_argv[] = {
            cc, "-o", "zig-wasm2c", "stage1/wasm2c.c", "-O2", "-std=c99", NULL,
        };
        print_and_run(child_argv);
    }
    {
        const char *child_argv[] = {
            "./zig-wasm2c", "stage1/zig1.wasm", "zig1.c", NULL,
        };
        print_and_run(child_argv);
    }
    {
        const char *child_argv[] = {
            cc, "-o", "zig1", "zig1.c", "stage1/wasi.c", "-std=c99", "-Os", "-fno-strict-aliasing", "-lm", NULL,
        };
        print_and_run(child_argv);
    }
    {
        FILE *f = fopen("config.zig", "wb");
        if (f == NULL)
            panic("unable to open config.zig for writing");

        const char *zig_version = "0.14.0-dev.bootstrap";

        int written = fprintf(f,
            "pub const have_llvm = false;\n"
            "pub const llvm_has_m68k = false;\n"
            "pub const llvm_has_csky = false;\n"
            "pub const llvm_has_arc = false;\n"
            "pub const llvm_has_xtensa = false;\n"
            "pub const version: [:0]const u8 = \"%s\";\n"
            "pub const semver = @import(\"std\").SemanticVersion.parse(version) catch unreachable;\n"
            "pub const enable_debug_extensions = false;\n"
            "pub const enable_logging = false;\n"
            "pub const enable_link_snapshots = false;\n"
            "pub const enable_tracy = false;\n"
            "pub const value_tracing = false;\n"
            "pub const skip_non_native = false;\n"
            "pub const debug_gpa = false;\n"
            "pub const dev = .core;\n"
            "pub const io_mode: enum { threaded, evented } = .threaded;\n"
            "pub const value_interpret_mode = .direct;\n"
        , zig_version);
        if (written < 100)
            panic("unable to write to config.zig file");
        if (fclose(f) != 0)
            panic("unable to finish writing to config.zig file");
    }

    {
        const char *child_argv[] = {
            "./zig1", "lib", "build-exe",
            "-ofmt=c", "-lc", "-OReleaseSmall",
            "--name", "zig2", "-femit-bin=zig2.c",
            "-target", host_triple,
            "--dep", "build_options",
            "--dep", "aro",
            "-Mroot=src/main.zig",
            "-Mbuild_options=config.zig",
            "-Maro=lib/compiler/aro/aro.zig",
            NULL,
        };
        print_and_run(child_argv);
    }

    {
        const char *child_argv[] = {
            "./zig1", "lib", "build-obj",
            "-ofmt=c", "-OReleaseSmall",
            "--name", "compiler_rt", "-femit-bin=compiler_rt.c",
            "-target", host_triple,
            "-Mroot=lib/compiler_rt.zig",
            NULL,
        };
        print_and_run(child_argv);
    }

    {
        const char *child_argv[] = {
            cc, "-o", "zig2", "zig2.c", "compiler_rt.c",
            "-std=c99", "-O2", "-fno-stack-protector",
            "-Istage1",
#if defined(__APPLE__)
            "-Wl,-stack_size,0x10000000",
#else
            "-Wl,-z,stack-size=0x10000000",
#endif
#if defined(__GNUC__)
            "-pthread",
#endif
            "-fno-strict-aliasing",
            workaround_gcc_sra_miscomp ? "-fno-tree-sra" : NULL,
            NULL,
        };
        print_and_run(child_argv);
    }
}
