#ifndef CSQLITE_SHIM_H
#define CSQLITE_SHIM_H

#include <sqlite3.h>

// Bridge SQLITE_TRANSIENT — Swift cannot import C macro-based function pointer constants
static inline sqlite3_destructor_type csqlite_TRANSIENT(void) {
    return SQLITE_TRANSIENT;
}

#endif
