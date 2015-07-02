############################### IPP ################################
set(IPP_FOUND)
set(ENV_SEARCH_PATH)

if(UNIX)
if(APPLE)
    set(ENV_SEARCH_PATH DYLD_LIBRARY_PATH)
else()
    set(ENV_SEARCH_PATH LD_LIBRARY_PATH)
endif()
endif()

foreach(v "7.0" "6.1" "6.0" "5.3" "5.2" "5.1")
    if(NOT IPP_FOUND)
        if(WIN32)
            find_path(IPP_PATH "ippi-${v}.dll"
                PATHS ${CMAKE_PROGRAM_PATH} ${CMAKE_SYSTEM_PROGRAM_PATH}
                DOC "The path to IPP dynamic libraries")
            if(NOT IPP_PATH)
                find_path(IPP_PATH "ippiem64t-${v}.dll"
                    PATHS ${CMAKE_PROGRAM_PATH} ${CMAKE_SYSTEM_PROGRAM_PATH}
                    DOC "The path to IPP dynamic libraries")
            endif()
        endif()
        if(UNIX)
            file(GLOB search_paths /opt/intel/ipp/*/*/sharedlib /usr/local/intel/ipp/*/*/sharedlib)

            find_path(IPP_PATH "libippi${CMAKE_SHARED_LIBRARY_SUFFIX}.${v}"
                PATHS ${CMAKE_LIBRARY_PATH} 
                ${CMAKE_SYSTEM_LIBRARY_PATH} 
                ${search_paths}
                ENV ${ENV_SEARCH_PATH}
                DOC "The path to IPP dynamic libraries")
            if(NOT IPP_PATH)
                find_path(IPP_PATH "libippiem64t${CMAKE_SHARED_LIBRARY_SUFFIX}.${v}"
                    PATHS ${CMAKE_LIBRARY_PATH} 
                    ${CMAKE_SYSTEM_LIBRARY_PATH} 
                    ${search_paths}
                    ENV ${ENV_SEARCH_PATH}
                    DOC "The path to IPP dynamic libraries")
            endif()
        endif()
        if(IPP_PATH)
            file(GLOB IPP_HDRS "${IPP_PATH}/../include")
            if(IPP_HDRS)
                set(IPP_FOUND TRUE)
            endif()
        endif()
    endif()
endforeach()

message(STATUS "IPP detected: ${IPP_FOUND}")

if(WIN32 AND NOT MSVC)
    set(IPP_FOUND)
endif()

set(USE_IPP ${IPP_FOUND} CACHE BOOL "Use IPP when available")

if(IPP_FOUND AND USE_IPP)
    message(STATUS "IPP found at: ${IPP_PATH}")

    add_definitions(-DHAVE_IPP)
    include_directories("${IPP_PATH}/../include")
    link_directories("${IPP_PATH}/../lib")
    
    file(GLOB em64t_files "${IPP_PATH}/../lib/*em64t*")
    set(IPP_ARCH)
    if(em64t_files)
        set(IPP_ARCH "em64t")
    endif()
    
    set(A ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(B ${IPP_ARCH}${CMAKE_STATIC_LIBRARY_SUFFIX})
    if(WIN32)
        set(L l)
    else()
        set(L)
    endif()
    set(IPP_LIBS ${A}ippjmerged${B} ${A}ippjemerged${B}
                 ${A}ippimerged${B} ${A}ippiemerged${B}
                 ${A}ippsmerged${B} ${A}ippsemerged${B}
                 #                 ${A}ippvmmerged${B} ${A}ippvmemerged${B}
                 #                 ${A}ippccmerged${B} ${A}ippccemerged${B}
                 #                 ${A}ippcvmerged${B} ${A}ippcvemerged${B}
                 ${A}ippcore${IPP_ARCH}${L}${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()
