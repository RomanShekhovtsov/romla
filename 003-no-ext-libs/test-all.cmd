call test-1-3.cmd %1 %2
IF NOT %ERRORLEVEL%==0 GOTO EXIT
call test-4-8.cmd %1 %2
:EXIT