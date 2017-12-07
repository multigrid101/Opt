I = terralib.includecstring [[
#include "/opt/intel/vtune_amplifier_xe_2017/include/ittnotify.h"
]]


-- set to true if itt analysis is desired but note that then executables can only
-- be executed via vtune and execution outside vtune will result in segfault.
-- If set to false, Any call to an __itt*** function will result in a no-op
-- local ALLOW_ITT = true
local ALLOW_ITT = false
-- TODO need to find out how to imitate the macros from the original C code, which
-- do not check for the value of a variable (here ALLOW_ITT) at *compile time*, but
-- instead check if the function pointers __itt_*__ptr...() are defined at *runtime*,
-- which makes the setting-by-hand  of a macro such as ALLOW_ITT unnecessary.


I.__itt_null = global(I.__itt_id, `I.__itt_id {0,0,0}, "__itt_null")

I.__itt_task_begin = macro(function(domain, arg2, arg3, name)
if ALLOW_ITT then
  return `I.__itt_task_begin_ptr__3_0(domain, arg2, arg3, name)
else
  return 0
end
end)


I.__itt_task_end = macro(function(domain)
if ALLOW_ITT then
  return `I.__itt_task_end_ptr__3_0(domain)
else
  return 0
end
end)


I.__itt_string_handle_create = macro(function(name)
if ALLOW_ITT then
  return `I.__itt_string_handle_create_ptr__3_0(name)
else
  return `nil
end
end)

I.__itt_domain_create = macro(function(name)
if ALLOW_ITT then
  return `I.__itt_domain_create_ptr__3_0(name)
else
  return `nil
end
end)


-------------------------------------------------------------------------------
return I
