/* See Copyright Notice in ../LICENSE.txt */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _WIN32
#error Currently Broken on Windows (Complain to Developers)
#include <io.h>
#include "ext/getopt.h"
#define isatty(x) _isatty(x)
#define NOMINMAX
#include <Windows.h>
#else
#ifdef __APPLE__
#include <mach-o/dyld.h> // for locating the executable
#include <sys/syslimits.h>
#define MAX_PATH_LEN PATH_MAX
#else // assuming Linux here
#include <linux/limits.h>
#define MAX_PATH_LEN PATH_MAX
#endif
#include <signal.h>
#include <getopt.h>
#include <unistd.h>
#include <libgen.h>
#endif
#include "terra.h"

struct ebb_Options {
  int uselegion;
  int usegpu;
  int ndebug;
  int legionspy;
  int legionprof;
  int logebb;
  int loglegion;
};

static void errquit(lua_State * L, const char *fmt, ...) {
    if (fmt) {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        fflush(stderr);
        va_end(args);
    }
    lua_close(L);
    terra_llvmshutdown();
    exit(1);
}
static void doerror(lua_State * L) {
    errquit(L, "%s\n",luaL_checkstring(L,-1));
}
const char * progname = NULL;
static void dotty (lua_State *L);
void parse_args(
  lua_State * L, int argc, char ** argv,
  terra_Options * options, ebb_Options * ebboptions,
  bool * interactive, int * begin_script
);
void check_legion_arg_consistency(ebb_Options * options);
static int getargs (lua_State *L, char **argv, int n);
static int docall (lua_State *L, int narg, int clear);

static void (*terratraceback)(void*);

#ifndef _WIN32
void sigsegv(int sig, siginfo_t *info, void * uap) {
    signal(sig,SIG_DFL); //reset signal to default, just in case traceback itself crashes
    terratraceback(uap);  //call terra's pretty traceback
    raise(sig);   //rethrow the signal to the default handler
}
void registerhandler() {
	struct sigaction sa;
	sa.sa_flags = SA_RESETHAND | SA_SIGINFO;
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = sigsegv;
	sigaction(SIGSEGV, &sa, NULL);
	sigaction(SIGILL, &sa, NULL);
}
#else
LONG WINAPI windowsexceptionhandler(EXCEPTION_POINTERS * ExceptionInfo) {
	terratraceback(ExceptionInfo->ContextRecord);
	return EXCEPTION_EXECUTE_HANDLER;
}
void registerhandler() {
	SetUnhandledExceptionFilter(windowsexceptionhandler);
}
#endif

char exec_dir_path_cache[MAX_PATH_LEN];
bool getexec_initialized = false;
char * getexec_dirpath() {
    if(getexec_initialized) return exec_dir_path_cache; 

    char        buf[MAX_PATH_LEN];
    int         maxbufsize  = MAX_PATH_LEN;
    int         bufsize     = maxbufsize;
#ifdef __APPLE__
    uint32_t bufsize_tmp = bufsize;
    if (_NSGetExecutablePath(buf, &bufsize_tmp) != 0) {
        return NULL; // error
    } else {
      bufsize = bufsize_tmp;
    }
#else
#ifdef _WIN32
    #error Implementation is Almost Certainly Broken here
    int retval = GetModuleFileName(NULL, buf, bufsize);
    if (retval == 0) {
        return NULL;
    } else {
        bufsize = retval;
    }
#else // Linux
    // NOTE: does not append a null byte
    int retval = readlink("/proc/self/exe", buf, bufsize);
    if (retval < 0) {
        return NULL;
    } else {
        bufsize = retval;
    }
#endif
    // BOTH Linux and Windows, not Apple
    // make sure the string fits and is null-terminated
    if (bufsize == maxbufsize) {
        return NULL;
    } else {
        buf[bufsize] = '\0'; // ensure there's a null terminator
    }
#endif

    char realbuf[MAX_PATH_LEN];
    // convert the path to the executable to a real path
    if(realpath(buf, realbuf) == NULL) {
        return NULL;
    }
    char *dirpath = dirname(realbuf);
    strncpy( exec_dir_path_cache, dirpath, MAX_PATH_LEN );

    getexec_initialized = true;
    return exec_dir_path_cache; // success
}

void setupcrashsignal(lua_State * L) {
    lua_getglobal(L, "terralib");
    lua_getfield(L, -1, "traceback");
    const void * tb = lua_topointer(L,-1);
    if(!tb)
        return; //debug not supported
    terratraceback = *(void(**)(void*))tb;
	registerhandler();
    lua_pop(L,2);
}

void setupebb(lua_State * L, ebb_Options * ebboptions) {
    char * bindir = getexec_dirpath();
    if(!bindir) {
        errquit(L, "ERROR: Failed to retreive path to this executable.\n");
    }

    char buffer[MAX_PATH_LEN]; // plenty of room
    size_t bufsize = MAX_PATH_LEN;

    // Make sure we can find the Terra files
    snprintf(buffer, bufsize,
      "package.terrapath = package.terrapath..';%s/../include/?.t;'",
      bindir);
    if (terra_dostring(L, buffer))
        doerror(L);

    if (ebboptions->uselegion) {
        // extend the Terra include path
        snprintf(buffer, bufsize,
          "terralib.includepath = terralib.includepath.."
          "';%s/../../legion/runtime;"
          "%s/../../legion/bindings/terra'",
          bindir, bindir);
        if (terra_dostring(L, buffer))
            doerror(L);

        // extend the Lua include path
        //snprintf(buffer, bufsize,
        //  "package.path = package.path.."
        //  "';%s/../../legion/bindings/terra/?.t'",
        //  bindir);
        //if (terra_dostring(L, buffer))
        //    doerror(L);

        // Link the Legion Shared Library into Terra
        if (ebboptions->ndebug) {
            snprintf(buffer, bufsize,"terralib.linklibrary("
              "'%s/../../legion/bindings/terra/liblegion_terra_release.so')",
              bindir);
            if (terra_dostring(L, buffer))
                doerror(L);
        } else {
            snprintf(buffer, bufsize,"terralib.linklibrary("
              "'%s/../../legion/bindings/terra/liblegion_terra_debug.so')",
              bindir);
            if (terra_dostring(L, buffer))
                doerror(L);
        }

        if (ebboptions->legionspy) {
            lua_pushboolean(L, true);
            lua_setglobal(L, "EBB_LEGION_USE_SPY");
        }
        if (ebboptions->legionprof) {
            lua_pushboolean(L, true);
            lua_setglobal(L, "EBB_LEGION_USE_PROF");
        }
	if (ebboptions->loglegion) {
            lua_pushboolean(L, true);
            lua_setglobal(L, "EBB_LEGION_LOG_LEGION");
	}
    }

    if (ebboptions->usegpu) {
        lua_pushboolean(L, true);
        lua_setglobal(L, "EBB_USE_GPU_SIGNAL");
    }
    if (ebboptions->logebb) {
        lua_pushboolean(L, true);
        lua_setglobal(L, "EBB_LEGION_LOG_EBB");
    }
}
int load_launchscript( lua_State * L, ebb_Options * ebboptions ) {
    char * bindir = getexec_dirpath();
    if(!bindir) {
        errquit(L, "ERROR: Failed to retreive path to this executable.\n");
    }

    char buffer[MAX_PATH_LEN]; // plenty of room
    size_t bufsize = MAX_PATH_LEN;

    if (ebboptions->uselegion) {
        snprintf(buffer, bufsize,
                 "%s/../include/ebb/src/launch_legion.t", bindir);
    } else {
        snprintf(buffer, bufsize,
                 "%s/../include/ebb/src/launch_script.t", bindir);
    }
    return terra_loadfile(L,buffer);
}

int main(int argc, char ** argv) {
    progname = argv[0];
    lua_State * L = luaL_newstate();
    luaL_openlibs(L);
    
    terra_Options terra_options;
    memset(&terra_options, 0, sizeof(terra_Options));

    ebb_Options ebboptions;
    memset(&ebboptions, 0, sizeof(ebb_Options));
    
    bool interactive = false;
    int scriptidx;

    parse_args(L,argc,argv,&terra_options,&ebboptions,&interactive,&scriptidx);
    // set some arguments by default
    terra_options.usemcjit = 1;
    // check other arguments
    check_legion_arg_consistency(&ebboptions);
    
    if(terra_initwithoptions(L, &terra_options))
        doerror(L);
    
    setupcrashsignal(L);
    setupebb(L, &ebboptions);
    
    if(scriptidx < argc) {
      const char * filename = argv[scriptidx];
      if(strcmp(filename,"-")) { // if not equal, then launch
        int narg = getargs(L, argv, scriptidx);
        lua_setglobal(L, "arg");
        if(load_launchscript(L,&ebboptions))
          doerror(L);
        lua_insert(L, -(narg + 1));
        if(docall(L,narg,0))
          doerror(L);
      }
    }
    
    if(isatty(0) && (interactive || scriptidx == argc)) {
        progname = NULL;
        dotty(L);
    }
    
    lua_close(L);
    terra_llvmshutdown();
    
    return 0;
}

static void print_welcome();
void usage() {
    print_welcome();
    printf(
      "ebb [OPTIONS] [source-file] [arguments-to-source-file]\n"
      "    -v=terra,ebb,legion enable verbose debugging output for one or more of terra, ebb and legion\n"
      "    -d enable debugging symbols\n"
      "    -h print this help message\n"
      "    -i enter the REPL after processing source files\n"
      "    -g run tasks on a gpu by default\n"
      "    -l enable Legion support\n"
      "    -n disable Legion debug mode\n"
      "    -s produce output for Legion spy\n"
      "    -p produce output for Legion prof\n"
      "    -  Execute stdin instead of script and stop parsing options\n");
}

void parse_args(
  lua_State * L, int  argc, char ** argv,
  terra_Options * options, ebb_Options * ebboptions,
  bool * interactive, int * begin_script
) {
    int ch;
    static struct option longopts[] = {
        { "help",           no_argument,          NULL,           'h' },
        { "verbose",        optional_argument,    NULL,           'v' },
        { "debugsymbols",   no_argument,          NULL,           'd' },
        { "interactive",    no_argument,          NULL,           'i' },
        { "gpu",            no_argument,          NULL,           'g' },
        { "legion",         no_argument,          NULL,           'l' },
        { "ndebug",         no_argument,          NULL,           'n' },
        { "spy",            no_argument,          NULL,           's' },
        { "prof",           no_argument,          NULL,           'p' },
        { NULL,             no_argument,          NULL,            0  }
    };
    /*  Parse commandline options  */
    opterr = 0;
    while ((ch = getopt_long(argc, argv, "+hv::idglnsp",
                             longopts, NULL)) != -1) {
        switch (ch) {
            case 'v':
	        if (optarg) {
		  if (strstr(optarg, "terra"))
		    options->verbose++;
		  if (strstr(optarg, "ebb"))
		    ebboptions->logebb = 1;
		  if (strstr(optarg, "legion"))
		    ebboptions->loglegion = 1;
		}
                break;
            case 'i':
                *interactive = true;
                break;
            case 'd':
                options->debug++;
                break;
            case 'g':
                ebboptions->usegpu = 1;
                break;
            case 'l':
                ebboptions->uselegion = 1;
                break;
            case 'n':
                ebboptions->ndebug = 1;
                break;
            case 's':
                ebboptions->legionspy = 1;
                break;
            case 'p':
                ebboptions->legionprof = 1;
                break;
            case ':':
            case 'h':
            default:
                usage();
                exit(-1);
                break;
        }
    }
    *begin_script = optind;
}
void check_legion_arg_consistency(ebb_Options * options) {
  if (options->legionspy) {
    if (!options->uselegion) {
      fprintf(stderr,
        "cannot generate Legion spy output when not running with Legion\n");
      exit(1);
    }
    if (options->ndebug) {
      fprintf(stderr, "Legion spy output can only be generated when running"
                      " in Legion debug mode\n");
      exit(1);
    }
  }
  if (options->legionprof) {
    if (!options->uselegion) {
      fprintf(stderr,
        "cannot generate Legion prof output when not running with Legion\n");
      exit(1);
    }
    if (!options->ndebug) {
      fprintf(stderr, "Legion prof output can only be generated when running"
                      " Legion not in debug mode\n");
      exit(1);
    }
  }
}

//this stuff is from lua's lua.c repl implementation:

#ifndef _WIN32
#include "linenoise.h"
#define lua_readline(L,b,p)    ((void)L, ((b)=linenoise(p)) != NULL)
#define lua_saveline(L,idx) \
    if (lua_strlen(L,idx) > 0)  /* non-empty line? */ \
      linenoiseHistoryAdd(lua_tostring(L, idx));  /* add it to history */
#define lua_freeline(L,b)    ((void)L, free(b))
#else
#define lua_readline(L,b,p)     \
        ((void)L, fputs(p, stdout), fflush(stdout),  /* show prompt */ \
        fgets(b, LUA_MAXINPUT, stdin) != NULL)  /* get line */
#define lua_saveline(L,idx)     { (void)L; (void)idx; }
#define lua_freeline(L,b)       { (void)L; (void)b; }
#endif

static void l_message (const char *pname, const char *msg) {
  if (pname) fprintf(stderr, "%s: ", pname);
  fprintf(stderr, "%s\n", msg);
  fflush(stderr);
}

static int report (lua_State *L, int status) {
  if (status && !lua_isnil(L, -1)) {
    const char *msg = lua_tostring(L, -1);
    if (msg == NULL) msg = "(error object is not a string)";
    l_message(progname, msg);
    lua_pop(L, 1);
  }
  return status;
}

static int incomplete (lua_State *L, int status) {
  if (status == LUA_ERRSYNTAX) {
    size_t lmsg;
    const char *msg = lua_tolstring(L, -1, &lmsg);
    const char *tp = msg + lmsg - (sizeof("'<eof>'") - 1);
    if (strstr(msg, "'<eof>'") == tp) {
      lua_pop(L, 1);
      return 1;
    }
  }
  return 0;  /* else... */
}

static int getargs (lua_State *L, char **argv, int n) {
  int narg;
  int i;
  int argc = 0;
  while (argv[argc]) argc++;  /* count total number of arguments */
  narg = argc - (n + 1);  /* number of arguments to the script */
  luaL_checkstack(L, narg + 3, "too many arguments to script");
  for (i=n+1; i < argc; i++)
    lua_pushstring(L, argv[i]);
  lua_createtable(L, narg, n + 1);
  for (i=0; i < argc; i++) {
    lua_pushstring(L, argv[i]);
    lua_rawseti(L, -2, i - n);
  }
  return narg;
}

#define LUA_MAXINPUT 512
#define LUA_PROMPT "> " 
#define LUA_PROMPT2 ">> "

static const char *get_prompt (lua_State *L, int firstline) {
  const char *p;
  lua_getfield(L, LUA_GLOBALSINDEX, firstline ? "_PROMPT" : "_PROMPT2");
  p = lua_tostring(L, -1);
  if (p == NULL) p = (firstline ? LUA_PROMPT : LUA_PROMPT2);
  lua_pop(L, 1);  /* remove global */
  return p;
}

static int pushline (lua_State *L, int firstline) {
  char buffer[LUA_MAXINPUT];
  char *b = buffer;
  size_t l;
  const char *prmt = get_prompt(L, firstline);
  if (lua_readline(L, b, prmt) == 0)
    return 0;  /* no input */
  l = strlen(b);
  if (l > 0 && b[l-1] == '\n')  /* line ends with newline? */
    b[l-1] = '\0';  /* remove it */
  if (firstline && b[0] == '=')  /* first line starts with `=' ? */
    lua_pushfstring(L, "return %s", b+1);  /* change it to `return' */
  else
    lua_pushstring(L, b);
  lua_saveline(L, -1);
  lua_freeline(L, b);
  return 1;
}

static int loadline (lua_State *L) {
  int status;
  lua_settop(L, 0);
  if (!pushline(L, 1))
    return -1;  /* no input */
  for (;;) {  /* repeat until gets a complete line */
    status = terra_loadbuffer(L, lua_tostring(L, 1), lua_strlen(L, 1), "stdin");
    if (!incomplete(L, status)) break;  /* cannot try to add lines? */
    if (!pushline(L, 0))  /* no more input? */
      return -1;
    lua_pushliteral(L, "\n");  /* add a new line... */
    lua_insert(L, -2);  /* ...between the two lines */
    lua_concat(L, 3);  /* join them */
  }
  lua_remove(L, 1);  /* remove line */
  return status;
}

static int traceback (lua_State *L) {
  if (!lua_isstring(L, 1))  /* 'message' not a string? */
    return 1;  /* keep it intact */
  lua_getfield(L, LUA_GLOBALSINDEX, "debug");
  if (!lua_istable(L, -1)) {
    lua_pop(L, 1);
    return 1;
  }
  lua_getfield(L, -1, "traceback");
  if (!lua_isfunction(L, -1)) {
    lua_pop(L, 2);
    return 1;
  }
  lua_pushvalue(L, 1);  /* pass error message */
  lua_pushinteger(L, 2);  /* skip this function and traceback */
  lua_call(L, 2, 1);  /* call debug.traceback */
  return 1;
}


static lua_State *globalL = NULL;

static void lstop (lua_State *L, lua_Debug *ar) {
  (void)ar;  /* unused arg. */
  lua_sethook(L, NULL, 0, 0);
  luaL_error(L, "interrupted!");
}

static void laction (int i) {
  signal(i, SIG_DFL); /* if another SIGINT happens before lstop,
                              terminate process (default action) */
  lua_sethook(globalL, lstop, LUA_MASKCALL | LUA_MASKRET | LUA_MASKCOUNT, 1);
}


static int docall (lua_State *L, int narg, int clear) {
  int status;
  globalL = L;
  int base = lua_gettop(L) - narg;  /* function index */
  lua_pushcfunction(L, traceback);  /* push traceback function */
  lua_insert(L, base);  /* put it under chunk and args */
  signal(SIGINT, laction);
  status = lua_pcall(L, narg, (clear ? 0 : LUA_MULTRET), base);
  signal(SIGINT, SIG_DFL);
  lua_remove(L, base);  /* remove traceback function */
  /* force a complete garbage collection in case of errors */
  if (status != 0) lua_gc(L, LUA_GCCOLLECT, 0);
  return status;
}
static void print_welcome() {
    printf("\n"
           "Liszt-Ebb -- A Language for Physical Simulation\n"
           "  (built w/ Terra)\n"
           "\n"
           "Stanford University\n"
           "chinmayee.shah@stanford.edu\n"
           "gilbert@gilbertbernstein.com\n"
           "\n");
}
static void dotty (lua_State *L) {
  int status;
  print_welcome();
  while ((status = loadline(L)) != -1) {
    if (status == 0) status = docall(L, 0, 0);
    report(L,status);
    if (status == 0 && lua_gettop(L) > 0) {  /* any result to print? */
      lua_getglobal(L, "print");
      lua_insert(L, 1);
      if (lua_pcall(L, lua_gettop(L)-1, 0, 0) != 0)
        lua_pushfstring(L,
                        "error calling " LUA_QL("print") " (%s)",
                        lua_tostring(L, -1));
        report(L,status);
    }
  }
  lua_settop(L, 0);  /* clear stack */
  fputs("\n", stdout);
  fflush(stdout);
}

#if 0
//a much simpler main function:
#include <stdio.h>
#include "terra.h"

static void doerror(lua_State * L) {
    printf("%s\n",luaL_checkstring(L,-1));
    exit(1);
}
int main(int argc, char ** argv) {
    lua_State * L = luaL_newstate();
    luaL_openlibs(L);
    if(terra_init(L))
        doerror(L);
    for(int i = 1; i < argc; i++)
        if(terra_dofile(L,argv[i]))
            doerror(L);
    return 0;
}
#endif
