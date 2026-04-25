"""
Tab completion scripts for tideon.ai CLI.

Usage:
    # Bash
    eval "$(teffgen --completion bash)"

    # Zsh
    eval "$(teffgen --completion zsh)"

    # Fish
    teffgen --completion fish | source
"""


def bash_completion() -> str:
    """Generate bash completion script."""
    return '''
_teffgen_completion() {
    local cur prev commands presets tools
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    commands="run chat serve config tools models examples health create-plugin presets"
    presets="math research coding general minimal"
    tools="calculator python_repl web_search code_executor file_operations bash json_tool datetime_tool text_processing url_fetch wikipedia agentic_search retrieval"

    case "${prev}" in
        teffgen)
            COMPREPLY=( $(compgen -W "${commands}" -- "${cur}") )
            return 0
            ;;
        --preset)
            COMPREPLY=( $(compgen -W "${presets}" -- "${cur}") )
            return 0
            ;;
        --tools|-t)
            COMPREPLY=( $(compgen -W "${tools}" -- "${cur}") )
            return 0
            ;;
        --mode)
            COMPREPLY=( $(compgen -W "auto single sub_agents" -- "${cur}") )
            return 0
            ;;
        run|chat)
            COMPREPLY=( $(compgen -W "--model --tools --preset --verbose --explain --stream --config --system-prompt --temperature --max-iterations --mode --output" -- "${cur}") )
            return 0
            ;;
        config)
            COMPREPLY=( $(compgen -W "show validate init" -- "${cur}") )
            return 0
            ;;
        tools)
            COMPREPLY=( $(compgen -W "list info test" -- "${cur}") )
            return 0
            ;;
        models)
            COMPREPLY=( $(compgen -W "list info" -- "${cur}") )
            return 0
            ;;
        examples)
            COMPREPLY=( $(compgen -W "list run" -- "${cur}") )
            return 0
            ;;
    esac

    if [[ "${cur}" == -* ]]; then
        COMPREPLY=( $(compgen -W "--help --version --verbose --log-file" -- "${cur}") )
    fi
}
complete -F _teffgen_completion teffgen
complete -F _teffgen_completion teffgen-agent
'''


def zsh_completion() -> str:
    """Generate zsh completion script."""
    return '''
#compdef teffgen teffgen-agent

_teffgen() {
    local -a commands presets tools
    commands=(
        'run:Run an agent with a task'
        'chat:Interactive chat mode'
        'serve:Start API server'
        'config:Configuration management'
        'tools:Tool management'
        'models:Model management'
        'examples:Run example scripts'
        'health:Check infrastructure health'
        'create-plugin:Generate plugin scaffold'
        'presets:List available presets'
    )
    presets=(math research coding general minimal)
    tools=(calculator python_repl web_search code_executor file_operations bash json_tool datetime_tool text_processing url_fetch wikipedia)

    _arguments -C \\
        '--version[Show version]' \\
        '--verbose[Verbose output]' \\
        '--log-file[Log file path]:file:_files' \\
        '1:command:->command' \\
        '*::arg:->args'

    case $state in
        command)
            _describe 'command' commands
            ;;
        args)
            case $words[1] in
                run)
                    _arguments \\
                        '--model[Model to use]:model:' \\
                        '--tools[Tools to enable]:tool:($tools)' \\
                        '--preset[Use preset config]:preset:($presets)' \\
                        '--verbose[Show execution stats]' \\
                        '--explain[Show tool reasoning]' \\
                        '--stream[Stream output]' \\
                        '--config[Config file]:file:_files' \\
                        '--temperature[Temperature]:temp:' \\
                        '--max-iterations[Max iterations]:n:' \\
                        '--output[Output file]:file:_files' \\
                        '1:task:'
                    ;;
            esac
            ;;
    esac
}

_teffgen "$@"
'''


def fish_completion() -> str:
    """Generate fish completion script."""
    return '''
complete -c teffgen -n "__fish_use_subcommand" -a "run" -d "Run an agent with a task"
complete -c teffgen -n "__fish_use_subcommand" -a "chat" -d "Interactive chat mode"
complete -c teffgen -n "__fish_use_subcommand" -a "serve" -d "Start API server"
complete -c teffgen -n "__fish_use_subcommand" -a "config" -d "Configuration management"
complete -c teffgen -n "__fish_use_subcommand" -a "tools" -d "Tool management"
complete -c teffgen -n "__fish_use_subcommand" -a "models" -d "Model management"
complete -c teffgen -n "__fish_use_subcommand" -a "health" -d "Check infrastructure health"
complete -c teffgen -n "__fish_use_subcommand" -a "create-plugin" -d "Generate plugin scaffold"
complete -c teffgen -n "__fish_use_subcommand" -a "presets" -d "List agent presets"

complete -c teffgen -n "__fish_seen_subcommand_from run" -l preset -xa "math research coding general minimal"
complete -c teffgen -n "__fish_seen_subcommand_from run" -l model -d "Model to use"
complete -c teffgen -n "__fish_seen_subcommand_from run" -l tools -xa "calculator python_repl web_search code_executor file_operations bash"
complete -c teffgen -n "__fish_seen_subcommand_from run" -l verbose -d "Show execution stats"
complete -c teffgen -n "__fish_seen_subcommand_from run" -l explain -d "Show tool reasoning"
complete -c teffgen -n "__fish_seen_subcommand_from run" -l stream -d "Stream output"
'''


def get_completion(shell: str) -> str:
    """Get completion script for the given shell."""
    generators = {
        "bash": bash_completion,
        "zsh": zsh_completion,
        "fish": fish_completion,
    }
    gen = generators.get(shell)
    if gen is None:
        raise ValueError(f"Unsupported shell '{shell}'. Supported: bash, zsh, fish")
    return gen()
