from mcp.server.fastmcp import FastMCP
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import asyncio
import sys
import tempfile
import json

# Create logger
logger = logging.getLogger(__name__)

# Global workflow directory mapping
workflow_dir_mapping = {}

# Create MCP service instance
mcp = FastMCP("GROMACS-VMD Service")

# Define dependencies
dependencies = []

# Import functional modules (using absolute imports)
from mcp_gmx_vmd.service import MCPService, SimulationParams
from mcp_gmx_vmd.gromacs import Context, run_gromacs_command
from mcp_gmx_vmd.models import (
    AnalysisParams, AnalysisResult, AnalysisType,
    CompleteSimulationParams, SimulationConfig,
    SimulationStatus, SimulationStep
)
from mcp_gmx_vmd.workflow_manager import WorkflowMetadata

# Get script directory as workspace root
SCRIPT_DIR = Path(__file__).parent.resolve()

# Debug info: output to stderr for viewing in Claude Desktop log
print(f"[DEBUG] Script directory: {SCRIPT_DIR}", file=sys.stderr, flush=True)
print(f"[DEBUG] Current working directory: {os.getcwd()}", file=sys.stderr, flush=True)

# Create service instance
service = MCPService(SCRIPT_DIR)

# Load workflow directory mapping
try:
    mapping_file = SCRIPT_DIR / ".mcp" / "workflow_dir_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, "r") as f:
            workflow_dir_mapping.update(json.load(f))
        logger.info(f"Load workflow directory mapping, {len(workflow_dir_mapping)}workflows total")
except Exception as e:
    logger.error(f"Load workflow directory mapping error: {e}")

# Add permission check and fix function
def ensure_workflow_directory_permissions(directory_path: Path) -> None:
    """Ensure workflow directory and subdirectories have correct permissions"""
    if not directory_path.exists():
        logger.warning(f"Workflow directory does not exist, cannot set permissions: {directory_path}")
        return
        
    try:
        import stat
        
        # Set main directory permissions
        os.chmod(directory_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        
        # Set subdirectory permissions
        for subdir in ["em", "nvt", "npt", "md"]:
            subdir_path = directory_path / subdir
            if subdir_path.exists():
                os.chmod(subdir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            else:
                # Create non-existent subdirectory and set permissions
                subdir_path.mkdir(parents=True, exist_ok=True)
                os.chmod(subdir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                
        logger.debug(f"Ensured workflow directory permissions: {directory_path}")
    except Exception as e:
        logger.warning(f"Failed to set directory permissions: {e}")

# Custom workflow directory retrieval function
def get_custom_workflow_directory(workflow_id: str) -> Optional[Path]:
    """Get workflow directory from mapping first, use default path if not found"""
    global workflow_dir_mapping
    
    # Get from mapping first
    if workflow_id in workflow_dir_mapping:
        custom_dir = Path(workflow_dir_mapping[workflow_id])
        if custom_dir.exists():
            logger.info(f"Found workflow in mapping {workflow_id} custom directory: {custom_dir}")
            # Ensure directory permissions are correct
            ensure_workflow_directory_permissions(custom_dir)
            return custom_dir
    
    # Fall back to default method
    workflow_dir = service.workflow_manager.get_workflow_directory(workflow_id)
    if workflow_dir:
        # Ensure default directory permissions are also correct
        ensure_workflow_directory_permissions(workflow_dir)
    return workflow_dir

# Load configuration from config file
config_file = SCRIPT_DIR / "config.json"
if config_file.exists():
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        vmd_config = config.get("vmd", {})
        gmx_config = config.get("gmx", {})
        logger.info(f"Load configuration from config file: {config_file}")
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        # Use default configuration
        vmd_config = {
            "vmd_path": "/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64",
            "structure_search_paths": [
                "/Users/tanqiong/01_myProject/30.vmd-mcp/00.mcp-test/01.mcp-vmd-gmx",
                "/Users/tanqiong/01_myProject/30.vmd-mcp/00.mcp-test/04.mcp-gmx-vmd_v4/mcp-gmx-vmd"
            ]
        }
        gmx_config = {
            "gmx_path": "gmx",
        }
else:
    logger.warning(f"Config file does not exist: {config_file}，Use default configuration")
    # Use default configuration
    vmd_config = {
        "vmd_path": "/Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64",
        "structure_search_paths": [
            "/Users/tanqiong/01_myProject/30.vmd-mcp/00.mcp-test/01.mcp-vmd-gmx",
            "/Users/tanqiong/01_myProject/30.vmd-mcp/00.mcp-test/04.mcp-gmx-vmd_v4/mcp-gmx-vmd"
        ]
    }
    gmx_config = {
        "gmx_path": "gmx",
    }

# Update service instance with configuration
service.vmd_manager.vmd_path = vmd_config["vmd_path"]
# Add structure file search path
for path in vmd_config["structure_search_paths"]:
    service.add_structure_search_path(path)

#====================
# Basic information
#====================

@mcp.resource("gmx-vmd://info")
async def get_info() -> dict:
    """Get service information"""
    return {
        "name": "MCP GROMACS-VMD Service",
        "version": "0.1.0",
        "description": "MCP service for molecular dynamics simulation and visualization"
    }

@mcp.resource("gmx-vmd://help")
async def get_help() -> str:
    """Get help information"""
    return service.get_workflow_help()

#====================
# Workflow management
#====================

@mcp.resource("gmx-vmd://workflows/create?name={name}&description={description}&params={params}")
async def create_workflow(name: str, description: str = "", params: Optional[Dict] = None) -> Dict:
    """Create new workflow"""
    workflow_params = CompleteSimulationParams(**params) if params else None
    workflow_id = service.create_workflow(name, description, workflow_params)
    return {"workflow_id": workflow_id, "success": True}

@mcp.resource("gmx-vmd://workflows/list")
async def list_workflows() -> List[Dict]:
    """List all workflows"""
    workflows = service.list_workflows()
    return [wf.to_dict() for wf in workflows]

@mcp.resource("gmx-vmd://workflows/get?workflow_id={workflow_id}")
async def get_workflow(workflow_id: str) -> Dict:
    """Get workflow details"""
    workflow = service.get_workflow(workflow_id)
    if workflow:
        return workflow.to_dict()
    return {"error": "Workflow does not exist", "workflow_id": workflow_id}

@mcp.resource("gmx-vmd://workflows/update?workflow_id={workflow_id}&name={name}&description={description}&status={status}&params={params}")
async def update_workflow(
    workflow_id: str, 
    name: Optional[str] = None, 
    description: Optional[str] = None,
    status: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Dict:
    """Update workflow"""
    status_obj = SimulationStatus(**status) if status else None
    params_obj = CompleteSimulationParams(**params) if params else None
    success = service.update_workflow(workflow_id, name, description, status_obj, params_obj)
    return {"success": success, "workflow_id": workflow_id}

@mcp.resource("gmx-vmd://workflows/delete?workflow_id={workflow_id}")
async def delete_workflow(workflow_id: str) -> Dict:
    """Delete workflow"""
    success = service.delete_workflow(workflow_id)
    return {"success": success, "workflow_id": workflow_id}

@mcp.resource("gmx-vmd://workflows/status?workflow_id={workflow_id}")
async def get_workflow_status(workflow_id: str) -> Dict:
    """Get workflow status"""
    status = service.get_workflow_status(workflow_id)
    if status:
        return status.dict()
    return {"error": "Cannot get workflow status", "workflow_id": workflow_id}

@mcp.resource("gmx-vmd://workflows/logs?workflow_id={workflow_id}")
async def get_workflow_logs(workflow_id: str) -> Dict:
    """Get workflow logs"""
    logs = service.get_workflow_logs(workflow_id)
    return {"logs": logs, "workflow_id": workflow_id}

@mcp.resource("gmx-vmd://workflows/checkpoints?workflow_id={workflow_id}")
async def get_workflow_checkpoints(workflow_id: str) -> Dict:
    """Get workflow checkpoints"""
    checkpoints = service.get_workflow_checkpoints(workflow_id)
    result = {}
    for step, files in checkpoints.items():
        result[step.value] = files
    return {"checkpoints": result, "workflow_id": workflow_id}

@mcp.resource("gmx-vmd://workflows/export?workflow_id={workflow_id}&output_file={output_file}")
async def export_workflow(workflow_id: str, output_file: str) -> Dict:
    """Export workflow"""
    success = service.export_workflow(workflow_id, output_file)
    return {"success": success, "workflow_id": workflow_id, "output_file": output_file}

@mcp.resource("gmx-vmd://workflows/import?input_file={input_file}")
async def import_workflow(input_file: str) -> Dict:
    """Import workflow"""
    workflow_id = service.import_workflow(input_file)
    if workflow_id:
        return {"success": True, "workflow_id": workflow_id}
    return {"success": False, "error": "Import workflow失败"}

#====================
# Simulation parameter management
#====================

@mcp.resource("gmx-vmd://parameters/validate?params={params}")
async def validate_parameters(params: Dict) -> Dict:
    """Validate simulation parameters"""
    params_obj = CompleteSimulationParams(**params)
    warnings = service.validate_parameters(params_obj)
    return {"warnings": warnings, "valid": not any(warnings.values())}

@mcp.resource("gmx-vmd://parameters/optimize?params={params}")
async def optimize_parameters(params: Dict) -> Dict:
    """Optimize simulation parameters"""
    params_obj = CompleteSimulationParams(**params)
    optimized_params, warnings = service.optimize_parameters(params_obj)
    return {
        "optimized_params": optimized_params.dict(),
        "warnings": warnings,
        "success": True
    }

#====================
# Trajectory analysis and visualization
#====================

@mcp.resource("gmx-vmd://analysis/trajectory?workflow_id={workflow_id}&params={params}")
async def analyze_trajectory(workflow_id: str, params: Dict) -> Dict:
    """Analyze trajectory"""
    # Log request information
    logger.info(f"Received trajectory analysis request:workflow_id={workflow_id}, params={params}")
    
    # Validate parameters
    errors = []
    
    # Check analysis type
    if "analysis_type" not in params:
        errors.append("Missing analysis type(analysis_type)parameter")
    elif params["analysis_type"].upper() == "RMSD" and params["analysis_type"] != "rmsd":
        errors.append("Analysis type error: RMSD needs to be specified in lowercase as rmsd")
    
    # Check structure file
    if "structure_file" not in params or not params["structure_file"]:
        errors.append("Missing field: need to provide reference structure_file parameter")
    
    # Check trajectory file
    if "trajectory_file" not in params or not params["trajectory_file"]:
        errors.append("Missing field: need to provide trajectory_file parameter")
    
    # If there are errors, return error message directly
    if errors:
        error_msg = "parameter验证失败: " + "; ".join(errors)
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "workflow_id": workflow_id
        }
    
    # 确保所有parameter都是正确的小写形式
    if "analysis_type" in params:
        params["analysis_type"] = params["analysis_type"].lower()
    
    try:
        # Get workflow directory
        workflow_dir = get_custom_workflow_directory(workflow_id)
        if not workflow_dir:
            error_msg = f"Cannot get workflow directory: {workflow_id}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "workflow_id": workflow_id
            }
        
        # Check file paths
        if "trajectory_file" in params and not os.path.isabs(params["trajectory_file"]):
            traj_path = os.path.join(workflow_dir, params["trajectory_file"])
            if not os.path.exists(traj_path):
                error_msg = f"Trajectory file does not exist: {traj_path}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "workflow_id": workflow_id
                }
        
        if "structure_file" in params and not os.path.isabs(params["structure_file"]):
            struct_path = os.path.join(workflow_dir, params["structure_file"])
            if not os.path.exists(struct_path):
                error_msg = f"Structure file does not exist: {struct_path}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "workflow_id": workflow_id
                }
        
        logger.info(f"parameter验证通过，准备创建AnalysisParams对象")
        
        # 创建analysisparameter对象
        try:
            analysis_params = AnalysisParams(**params)
            logger.info(f"Successfully created AnalysisParams object: {analysis_params}")
        except Exception as e:
            error_msg = f"创建analysisparameter对象失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False, 
                "error": error_msg,
                "workflow_id": workflow_id
            }
        
        # Execute analysis, passing custom workflow directory
        logger.info(f"Starting to call service.analyze_trajectory")
        result = await service.analyze_trajectory(workflow_id, analysis_params, workflow_dir)
        
        if result:
            logger.info(f"Analysis completed successfully, returning result")
            return {
                "success": True,
                "result": result.dict(),
                "workflow_id": workflow_id
            }
            
        logger.error("Trajectory analysis failed, service.analyze_trajectory returned None")
        return {
            "success": False, 
            "error": "Trajectory analysis failed，请检查parameter和日志", 
            "workflow_id": workflow_id
        }
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Analyze trajectory时发生错误: {error_msg}")
        logger.error(f"Exception stack trace: {tb}")
        
        return {
            "success": False,
            "error": f"Analyze trajectory时发生错误: {error_msg}",
            "traceback": tb,
            "workflow_id": workflow_id
        }

@mcp.resource("gmx-vmd://visualization/apply-template?workflow_id={workflow_id}&template_name={template_name}&params={params}")
async def apply_vmd_template(workflow_id: str, template_name: str, params: Optional[Dict] = None) -> Dict:
    """Apply VMD template"""
    # Get custom workflow directory
    workflow_dir = get_custom_workflow_directory(workflow_id)
    if not workflow_dir:
        return {
            "success": False,
            "error": f"Cannot get workflow directory: {workflow_id}",
            "workflow_id": workflow_id,
            "template": template_name
        }
    
    # Call service method, passing custom workflow directory
    success = service.apply_vmd_template(workflow_id, template_name, params, workflow_dir)
    return {
        "success": success,
        "workflow_id": workflow_id,
        "template": template_name
    }

@mcp.resource("gmx-vmd://visualization/templates")
async def get_available_templates() -> Dict:
    """Get available VMD templates"""
    templates = service.get_available_templates()
    return {"templates": templates}

#====================
# GROMACS command execution
#====================

@mcp.resource("gmx-vmd://gromacs/execute?workflow_id={workflow_id}&command={command}&args={args}&input_data={input_data}")
async def execute_gromacs_command(workflow_id: str, command: str, args: List[str] = None, input_data: Optional[str] = None) -> Dict:
    """Execute GROMACS command"""
    workflow_dir = get_custom_workflow_directory(workflow_id)
    if not workflow_dir:
        return {"success": False, "error": "Workflow directory does not exist", "workflow_id": workflow_id}
    
    # Check if working directory exists
    if not os.path.isdir(workflow_dir):
        return {"success": False, "error": f"Workflow directory does not exist: {workflow_dir}", "workflow_id": workflow_id}
    
    # 查找命令parameter中的File路径，并转换for绝对路径
    if args:
        for i, arg in enumerate(args):
            # Process"-"开头的选项后面的parameter
            if arg.startswith("-") and i + 1 < len(args):
                next_arg = args[i + 1]
                if not next_arg.startswith("-"):
                    # 判断这个选项是否是File路径parameter
                    file_options = ['-c', '-s', '-f', '-r', '-t', '-n', '-o', '-e', '-g', '-cpi']
                    if arg in file_options:
                        # Convert file path to absolute path
                        if not os.path.isabs(next_arg):
                            # If path contains slash, use directly; otherwise add directory prefix
                            if '/' in next_arg:
                                args[i + 1] = str(workflow_dir / next_arg)
                            else:
                                # Special handling for filenames without directory separator
                                # For some options, add specific subdirectory
                                if arg in ['-t', '-cpi'] and next_arg.startswith('nvt'):
                                    # For checkpoint files, like nvt.cpt should be in nvt/ directory
                                    args[i + 1] = str(workflow_dir / "nvt" / next_arg)
                                elif arg in ['-t', '-cpi'] and next_arg.startswith('npt'):
                                    # 对于checkpointFile，如npt.cpt应该在npt/目录下
                                    args[i + 1] = str(workflow_dir / "npt" / next_arg)
                                else:
                                    # Other files default to working directory
                                    args[i + 1] = str(workflow_dir / next_arg)
            
            # Handle -deffnm option, followed by file prefix without extension
            elif arg == "-deffnm" and i + 1 < len(args):
                next_arg = args[i + 1]
                if not os.path.isabs(next_arg):
                    # 如果路径中包含斜杠，直接use；否则添加Working directoryprefix
                    args[i + 1] = str(workflow_dir / next_arg)
    
    # 输出实际use的parameter（调试用）
    logger.debug(f"处理后的命令parameter: {args}")
    
    # Validate if input files exist
    if args:
        missing_files = []
        for i, arg in enumerate(args):
            if arg in ['-c', '-s', '-f', '-r', '-t', '-n'] and i + 1 < len(args):
                file_path = args[i + 1]
                if not os.path.isabs(file_path):
                    file_path = os.path.join(workflow_dir, file_path)
                if not os.path.exists(file_path):
                    missing_files.append(f"File '{args[i + 1]}' does not exist")
        
        if missing_files:
            return {
                "success": False,
                "error": f"输入Filedoes not exist: {', '.join(missing_files)}",
                "workflow_id": workflow_id,
                "command": command
            }
    
    # Special handling for GROMACS command format
    # GROMACS 5+ use "gmx <command>" format，而旧版直接use命令名
    gmx_cmd = gmx_config["gmx_path"]  # usually "gmx"
    actual_command = command
    actual_args = args or []
    
    # Check if command already contains gmx prefix
    if command.startswith("gmx "):
        # 命令已包含prefix，分离出实际命令
        parts = command.split(" ", 1)
        actual_command = parts[1] if len(parts) > 1 else ""
    elif not command.startswith("gmx") and gmx_cmd == "gmx":
        # 如果命令不包含prefix，且配置foruseprefix，则不做特殊处理
        # This case will be handled by run_gromacs_command correctly handled
        pass
    
    # 确保输出目录存在
    output_path = None
    if args:
        for i, arg in enumerate(args):
            if arg == '-o' and i + 1 < len(args):
                output_path = args[i + 1]
                if not os.path.isabs(output_path):
                    output_path = os.path.join(workflow_dir, output_path)
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
    
    # 设置执行上下文并执行命令
    ctx = Context(working_dir=workflow_dir, gmx_path=gmx_config["gmx_path"])
    
    # 记录实际执行的命令（用于调试）
    logger.info(f"Execute GROMACS command: {gmx_cmd} {actual_command} {' '.join(str(a) for a in actual_args)}")
    
    result = await run_gromacs_command(ctx, actual_command, actual_args, input_data)
    
    # 构建完整命令字符串，用于显示
    full_command = f"{gmx_cmd} {actual_command} {' '.join(str(a) for a in actual_args)}"
    
    return {
        "success": result.success,
        "output": result.stdout,  # 返回标准输出
        "error": result.stderr,   # 返回错误输出
        "return_code": result.return_code,
        "workflow_id": workflow_id,
        "command": command,
        "full_command": full_command,
        "debug_info": {
            "working_dir": str(workflow_dir),
            "command_executed": f"{gmx_cmd} {actual_command}",
            "args": [str(a) for a in actual_args]
        }
    }

#====================
# MCP工具定义
#====================

# Basic information工具
@mcp.tool("get_service_info")
async def get_info_tool() -> Dict:
    """获取GMX-VMD服务的Basic information，包括名称、版本和描述"""
    return await get_info()

@mcp.tool("get_help")
async def get_help_tool() -> str:
    """获取GMX-VMD服务的use帮助和工作流程指南"""
    return await get_help()

# Workflow management工具
@mcp.tool("create_workflow")
async def create_workflow_tool(name: str, description: str = "", params: Optional[Dict] = None, workspace_dir: Optional[str] = None) -> Dict:
    """创建一个新的Molecular dynamics simulation workflow
    
    Args:
        name: 工作流程名称
        description: 工作流程描述
        params: 模拟parameter（可选）
        workspace_dir: 工作流的Working directory（可选，默认forMCP服务的Working directory）
    """
    global workflow_dir_mapping
    
    if workspace_dir:
        # 如果指定了workspace_dir，创建一个临时的WorkflowManager
        from mcp_gmx_vmd.workflow_manager import WorkflowManager
        from mcp_gmx_vmd.models import CompleteSimulationParams
        
        # 确保目录存在并设置正确的权限
        workspace_path = Path(workspace_dir)
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # 设置目录权限for777，确保所有用户都有完全访问权限
        try:
            import stat
            os.chmod(workspace_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 等同于 chmod 777
            logger.info(f"已设置目录权限: {workspace_path}")
        except Exception as e:
            logger.warning(f"Failed to set directory permissions: {e}")
        
        # 创建临时工作流管理器
        temp_manager = WorkflowManager(workspace_path)
        
        # 解析parameter（如果有）
        workflow_params = CompleteSimulationParams(**params) if params else None
        
        # 创建工作流
        workflow_id = temp_manager.create_workflow(name, description, workflow_params)
        
        # Get workflow directory并设置权限
        workflow_dir = workspace_path / workflow_id
        if workflow_dir.exists():
            try:
                import stat
                os.chmod(workflow_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 等同于 chmod 777
                logger.info(f"已设置工作流目录权限: {workflow_dir}")
                
                # 确保子目录也有正确的权限
                for subdir in ["em", "nvt", "npt", "md"]:
                    subdir_path = workflow_dir / subdir
                    subdir_path.mkdir(parents=True, exist_ok=True)
                    os.chmod(subdir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except Exception as e:
                logger.warning(f"设置工作流目录权限失败: {e}")
        
        # 将此工作流元数据复制到主服务的工作流管理器中
        metadata = temp_manager.get_workflow(workflow_id)
        if metadata:
            service.workflow_manager._save_metadata(metadata)
            
        # 记录工作流目录映射
        workflow_dir_mapping[workflow_id] = str(workflow_dir)
        logger.info(f"已记录工作流 {workflow_id} custom directory: {workflow_dir_mapping[workflow_id]}")
        
        # 保存工作流目录映射到File，确保服务重启后仍能找到
        try:
            mapping_file = Path(os.getcwd()) / ".mcp" / "workflow_dir_mapping.json"
            mapping_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 读取现有映射（如果存在）
            existing_mapping = {}
            if mapping_file.exists():
                with open(mapping_file, "r") as f:
                    existing_mapping = json.load(f)
            
            # 更新映射
            existing_mapping[workflow_id] = str(workflow_dir)
            
            # 保存更新后的映射
            with open(mapping_file, "w") as f:
                json.dump(existing_mapping, f, indent=4)
                
            logger.info(f"工作流目录映射已保存到: {mapping_file}")
        except Exception as e:
            logger.error(f"保存工作流目录映射时出错: {e}")
        
        return {"workflow_id": workflow_id, "success": True, "workspace_dir": str(workflow_dir)}
    else:
        # use默认工作流管理器
        return await create_workflow(name, description, params)

@mcp.tool("list_workflows")
async def list_workflows_tool() -> List[Dict]:
    """获取所有已创建的工作流程列表"""
    return await list_workflows()

@mcp.tool("get_workflow")
async def get_workflow_tool(workflow_id: str) -> Dict:
    """获取指定工作流程的详细信息
    
    Args:
        workflow_id: Workflow ID
    """
    return await get_workflow(workflow_id)

@mcp.tool("update_workflow")
async def update_workflow_tool(workflow_id: str, name: Optional[str] = None, 
                             description: Optional[str] = None,
                             status: Optional[Dict] = None,
                             params: Optional[Dict] = None) -> Dict:
    """更新现有工作流程的信息
    
    Args:
        workflow_id: Workflow ID
        name: 新的工作流程名称（可选）
        description: 新的工作流程描述（可选）
        status: 工作流程状态信息（可选）
        params: 模拟parameter（可选）
    """
    return await update_workflow(workflow_id, name, description, status, params)

@mcp.tool("delete_workflow")
async def delete_workflow_tool(workflow_id: str) -> Dict:
    """删除指定的工作流程
    
    Args:
        workflow_id: 要删除的Workflow ID
    """
    return await delete_workflow(workflow_id)

@mcp.tool("get_workflow_status")
async def get_workflow_status_tool(workflow_id: str) -> Dict:
    """获取指定工作流程的当前运行状态
    
    Args:
        workflow_id: Workflow ID
    """
    return await get_workflow_status(workflow_id)

@mcp.tool("get_workflow_logs")
async def get_workflow_logs_tool(workflow_id: str) -> Dict:
    """获取指定工作流程的运行日志
    
    Args:
        workflow_id: Workflow ID
    """
    return await get_workflow_logs(workflow_id)

@mcp.tool("get_workflow_checkpoints")
async def get_workflow_checkpoints_tool(workflow_id: str) -> Dict:
    """获取指定工作流程的所有检查点
    
    Args:
        workflow_id: Workflow ID
    """
    return await get_workflow_checkpoints(workflow_id)

@mcp.tool("export_workflow")
async def export_workflow_tool(workflow_id: str, output_file: str) -> Dict:
    """将指定工作流程导出到File
    
    Args:
        workflow_id: Workflow ID
        output_file: 导出File路径
    """
    return await export_workflow(workflow_id, output_file)

@mcp.tool("import_workflow")
async def import_workflow_tool(input_file: str) -> Dict:
    """从FileImport workflow
    
    Args:
        input_file: 导入File路径
    """
    return await import_workflow(input_file)

# parameter管理工具
@mcp.tool("validate_parameters")
async def validate_parameters_tool(params: Dict) -> Dict:
    """验证分子动Force学模拟parameter的有效性
    
    Args:
        params: 模拟parameter
    """
    return await validate_parameters(params)

@mcp.tool("optimize_parameters")
async def optimize_parameters_tool(params: Dict) -> Dict:
    """优化分子动Force学模拟parameter，提高模拟效率和稳定性
    
    Args:
        params: 原始模拟parameter
    """
    return await optimize_parameters(params)

# Trajectory analysis工具
@mcp.tool("analyze_trajectory")
async def analyze_trajectory_tool(workflow_id: str, params: Dict) -> Dict:
    """analysis模拟轨迹数据，提取结构和动Force学信息
    
    Args:
        workflow_id: Workflow ID
        params: analysisparameter
    """
    try:
        logger.info(f"开始Trajectory analysis，工作流ID: {workflow_id}, parameter: {params}")
        
        # 检查工作流是否存在
        workflow_dir = get_custom_workflow_directory(workflow_id)
        if not workflow_dir:
            logger.error(f"工作流目录does not exist: {workflow_id}")
            return {
                "success": False,
                "error": f"工作流目录does not exist: {workflow_id}"
            }
        
        # Check file paths
        if "trajectory_file" in params and not os.path.isabs(params["trajectory_file"]):
            trajectory_file = os.path.join(workflow_dir, params["trajectory_file"])
            if not os.path.exists(trajectory_file):
                logger.error(f"Trajectory file does not exist: {trajectory_file}")
                return {
                    "success": False,
                    "error": f"Trajectory file does not exist: {params['trajectory_file']}"
                }
            # 更新for相对路径，防止路径问题
            params["trajectory_file"] = os.path.relpath(trajectory_file, workflow_dir)
        
        if "structure_file" in params and not os.path.isabs(params["structure_file"]):
            structure_file = os.path.join(workflow_dir, params["structure_file"])
            if not os.path.exists(structure_file):
                logger.error(f"Structure file does not exist: {structure_file}")
                return {
                    "success": False,
                    "error": f"Structure file does not exist: {params['structure_file']}"
                }
            # 更新for相对路径，防止路径问题
            params["structure_file"] = os.path.relpath(structure_file, workflow_dir)
        
        logger.info(f"File检查通过，准备执行analysis: trajectory_file={params.get('trajectory_file')}, structure_file={params.get('structure_file')}")
        
        # 调用API函数执行analysis
        result = await analyze_trajectory(workflow_id, params)
        
        # 记录结果
        if not result.get("success", False):
            logger.error(f"Trajectory analysis failed: {result.get('error', '未知错误')}")
        else:
            logger.info(f"Trajectory analysis成功完成")
            
        return result
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Trajectory analysis过程中发生异常: {error_msg}")
        logger.error(f"Exception stack trace: {tb}")
        
        return {
            "success": False,
            "error": f"Analyze trajectory时发生错误: {error_msg}",
            "details": tb
        }

# Visualization工具
@mcp.tool("apply_vmd_template")
async def apply_vmd_template_tool(workflow_id: str, template_name: str, params: Optional[Dict] = None) -> Dict:
    """应用预定义的VMDVisualization模板
    
    Args:
        workflow_id: Workflow ID
        template_name: 模板名称
        params: 模板parameter（可选）
    """
    return await apply_vmd_template(workflow_id, template_name, params)

@mcp.tool("get_available_templates")
async def get_available_templates_tool() -> Dict:
    """获取所有可用的VMDVisualization模板列表"""
    return await get_available_templates()

# GROMACS command execution工具
@mcp.tool("execute_gromacs_command")
async def execute_gromacs_command_tool(workflow_id: str, command: str, args: List[str] = None, input_data: Optional[str] = None) -> Dict:
    """Execute GROMACS command行工具
    
    Args:
        workflow_id: Workflow ID
        command: GROMACS命令名称
        args: 命令parameter列表（可选）
        input_data: 标准输入数据（可选，用于需要交互式输入的命令，如genion）
        
    Returns:
        Dict: 包含命令执行结果、标准输出和错误输出的字典
    """
    result = await execute_gromacs_command(workflow_id, command, args, input_data)
    # 确保将完整的标准输出和错误信息返回给用户
    return {
        "success": result["success"],
        "output": result["output"],    # 完整的命令标准输出
        "error": result["error"],      # 完整的命令错误输出
        "return_code": result["return_code"],
        "workflow_id": workflow_id,
        "command": command,
        "full_command": result.get("full_command", ""),
        "debug_info": result.get("debug_info", {})
    }

@mcp.tool("execute_gromacs_sequence")
async def execute_gromacs_command_sequence_tool(workflow_id: str, commands: List[Dict]) -> Dict:
    """执行一系列GROMACS命令
    
    连续执行多个GROMACS命令，适用于完成分子动Force学模拟的完整流程。
    每个命令会等待前一个命令完成后再执行。
    
    Args:
        workflow_id: Workflow ID
        commands: 命令列表，每个命令包含command、args和可选的input_data
        
    Returns:
        Dict: 包含所有命令执行结果的字典
    """
    results = []
    workflow_dir = get_custom_workflow_directory(workflow_id)
    
    if not workflow_dir:
        return {
            "success": False, 
            "error": "Workflow directory does not exist", 
            "workflow_id": workflow_id
        }
    
    # 确保Working directory存在
    if not os.path.isdir(workflow_dir):
        error_msg = f"Workflow directory does not exist或无法访问: {workflow_dir}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "workflow_id": workflow_id
        }
    
    # 确保Working directory有正确的权限
    ensure_workflow_directory_permissions(workflow_dir)
    
    # 记录Basic information
    logger.info(f"开始执行命令序列，共{len(commands)}个命令，工作流ID: {workflow_id}")
    logger.info(f"Working directory: {workflow_dir}")
    
    for idx, cmd_info in enumerate(commands):
        command = cmd_info.get("command")
        args = cmd_info.get("args", [])
        input_data = cmd_info.get("input_data")
        step = cmd_info.get("step", f"Step{idx+1}")
        
        logger.info(f"执行命令序列 - {step}: {command} {' '.join(str(a) for a in (args or []))}")
        
        # 执行单个命令
        result = await execute_gromacs_command(workflow_id, command, args, input_data)
        
        # 添加Step信息
        result["step"] = step
        results.append(result)
        
        # 如果命令失败，打印详细错误信息并停止执行后续命令
        if not result["success"]:
            error_msg = result.get("error", "未知错误")
            output_msg = result.get("output", "")
            logger.error(f"命令序列在Step {step} 失败")
            logger.error(f"错误信息: {error_msg}")
            logger.error(f"命令输出: {output_msg}")
            
            # 获取更多调试信息
            debug_info = result.get("debug_info", {})
            if debug_info:
                logger.error(f"调试信息: {debug_info}")
                
            # 检查输入File是否存在
            if args:
                for i, arg in enumerate(args):
                    if arg in ['-c', '-s', '-f', '-r', '-t', '-n'] and i + 1 < len(args):
                        file_path = args[i + 1]
                        if not os.path.isabs(file_path):
                            file_path = os.path.join(workflow_dir, file_path)
                        if not os.path.exists(file_path):
                            logger.error(f"Filedoes not exist: {file_path}")
            
            break
    
    # 计算整体成功/失败状态
    all_success = all(r["success"] for r in results)
    
    # 汇总错误信息
    all_errors = []
    for r in results:
        if not r.get("success", False) and r.get("error"):
            all_errors.append(f"{r.get('step', '未知Step')}: {r.get('error')}")
    
    return {
        "success": all_success,
        "results": results,
        "workflow_id": workflow_id,
        "completed_steps": len(results),
        "total_steps": len(commands),
        "error": "; ".join(all_errors) if all_errors else None
    }

@mcp.tool("run_md_simulation_stage")
async def run_md_simulation_stage_tool(workflow_id: str, stage: str) -> Dict:
    """运行指定阶段的分子动Force学模拟
    
    执行分子动Force学模拟的特定阶段，如Energy minimization、NVT平衡或NPT平衡。
    
    Args:
        workflow_id: Workflow ID
        stage: 模拟阶段名称 (minimization, nvt, npt, production)
        
    Returns:
        Dict: 包含模拟执行结果的字典
    """
    # Get workflow information
    workflow = service.get_workflow(workflow_id)
    if not workflow:
        return {"success": False, "error": f"Workflow does not exist: {workflow_id}"}
        
    # 获取Working directory
    workflow_dir = get_custom_workflow_directory(workflow_id)
    if not workflow_dir:
        return {"success": False, "error": f"Cannot get workflow directory: {workflow_id}"}
        
    # 确保所需目录存在
    for subdir in ["em", "nvt", "npt", "md"]:
        os.makedirs(os.path.join(workflow_dir, subdir), exist_ok=True)
    
    # 检查前置Step是否已完成
    if stage != "minimization":
        # 检查Energy minimization结果
        em_gro = os.path.join(workflow_dir, "em", "em.gro")
        if not os.path.exists(em_gro):
            return {
                "success": False, 
                "error": f"Energy minimization结果Filedoes not exist: {em_gro}，请先运行Energy minimizationStep"
            }
    
    if stage in ["npt", "production"]:
        # 检查NVT平衡结果
        nvt_gro = os.path.join(workflow_dir, "nvt", "nvt.gro")
        nvt_cpt = os.path.join(workflow_dir, "nvt", "nvt.cpt")
        if not os.path.exists(nvt_gro) or not os.path.exists(nvt_cpt):
            return {
                "success": False, 
                "error": f"NVT平衡结果Filedoes not exist，请先Run NVT equilibrationStep"
            }
    
    if stage == "production":
        # 检查NPT平衡结果
        npt_gro = os.path.join(workflow_dir, "npt", "npt.gro")
        npt_cpt = os.path.join(workflow_dir, "npt", "npt.cpt")
        if not os.path.exists(npt_gro) or not os.path.exists(npt_cpt):
            return {
                "success": False, 
                "error": f"NPT平衡结果Filedoes not exist，请先Run NPT equilibrationStep"
            }
    
    # 根据阶段名称准备不同的命令
    commands = []
    
    if stage == "minimization":
        # Energy minimization阶段
        commands = [
            {
                "step": "Energy minimization准备",
                "command": "grompp",
                "args": [
                    "-f", str(workflow_dir / "em/em.mdp"),
                    "-c", str(workflow_dir / "solv_ions.gro"),
                    "-p", str(workflow_dir / "topol.top"),
                    "-o", str(workflow_dir / "em/em.tpr")
                ]
            },
            {
                "step": "运行Energy minimization",
                "command": "mdrun",
                "args": [
                    "-v",
                    "-s", str(workflow_dir / "em/em.tpr"),
                    "-deffnm", str(workflow_dir / "em/em")
                ]
            }
        ]
    elif stage == "nvt":
        # NVT平衡阶段
        commands = [
            {
                "step": "NVT equilibration preparation",
                "command": "grompp",
                "args": [
                    "-f", str(workflow_dir / "nvt/nvt.mdp"),
                    "-c", str(workflow_dir / "em/em.gro"), 
                    "-r", str(workflow_dir / "em/em.gro"),  # 约束参考坐标
                    "-p", str(workflow_dir / "topol.top"),
                    "-o", str(workflow_dir / "nvt/nvt.tpr"),
                    "-maxwarn", "1"  # 允许一些警告
                ]
            },
            {
                "step": "Run NVT equilibration",
                "command": "mdrun",
                "args": [
                    "-v",
                    "-s", str(workflow_dir / "nvt/nvt.tpr"),
                    "-deffnm", str(workflow_dir / "nvt/nvt")
                ]
            }
        ]
    elif stage == "npt":
        # NPT平衡阶段
        commands = [
            {
                "step": "NPT equilibration preparation",
                "command": "grompp",
                "args": [
                    "-f", str(workflow_dir / "npt/npt.mdp"),
                    "-c", str(workflow_dir / "nvt/nvt.gro"),
                    "-r", str(workflow_dir / "nvt/nvt.gro"),
                    "-t", str(workflow_dir / "nvt/nvt.cpt"),
                    "-p", str(workflow_dir / "topol.top"),
                    "-o", str(workflow_dir / "npt/npt.tpr"),
                    "-maxwarn", "1"  # 允许一些警告
                ]
            },
            {
                "step": "Run NPT equilibration",
                "command": "mdrun",
                "args": [
                    "-v",
                    "-s", str(workflow_dir / "npt/npt.tpr"),
                    "-deffnm", str(workflow_dir / "npt/npt")
                ]
            }
        ]
    elif stage == "production":
        # Production simulation阶段
        commands = [
            {
                "step": "Production simulation准备",
                "command": "grompp",
                "args": [
                    "-f", str(workflow_dir / "md/md.mdp"),
                    "-c", str(workflow_dir / "npt/npt.gro"),
                    "-t", str(workflow_dir / "npt/npt.cpt"),
                    "-p", str(workflow_dir / "topol.top"),
                    "-o", str(workflow_dir / "md/md.tpr"),
                    "-maxwarn", "1"  # 允许一些警告
                ]
            },
            {
                "step": "运行Production simulation",
                "command": "mdrun",
                "args": [
                    "-v",
                    "-s", str(workflow_dir / "md/md.tpr"),
                    "-deffnm", str(workflow_dir / "md/md")
                ]
            }
        ]
    else:
        return {"success": False, "error": f"未知的模拟阶段: {stage}"}
    
    # 记录阶段执行的开始
    logger.info(f"开始执行{stage}阶段模拟，工作流ID: {workflow_id}")
        
    # 执行命令序列
    result = await execute_gromacs_command_sequence_tool(workflow_id, commands)
    
    # 添加调试信息
    if not result["success"]:
        logger.error(f"{stage}阶段执行失败: {result.get('error', '未知错误')}")
        # 检查是否有具体错误信息
        if "results" in result:
            for cmd_result in result["results"]:
                if not cmd_result.get("success", False):
                    step = cmd_result.get("step", "未知Step")
                    error = cmd_result.get("error", "未知错误")
                    logger.error(f"Step {step} 失败: {error}")
    else:
        logger.info(f"{stage}阶段执行成功")
    
    return result

# VMD相关工具
@mcp.tool("launch_vmd_gui")
async def launch_vmd_gui_tool(structure_file: Optional[str] = None, trajectory_file: Optional[str] = None) -> Dict:
    """启动VMD的图形界面并加载分子File
    
    启动VMD分子Visualization程序的图形用户界面，可以同时加载结构File和轨迹File。
    
    Args:
        structure_file: 可选的结构File路径，如.gro、.pdb等
        trajectory_file: 可选的轨迹File路径，如.xtc、.trr等
    
    Returns:
        Dict: 包含进程ID和启动状态的字典
    """
    # 检查File是否存在
    if structure_file and not os.path.exists(structure_file):
        return {
            "success": False,
            "error": f"Structure file does not exist: {structure_file}"
        }
    
    if trajectory_file and not os.path.exists(trajectory_file):
        return {
            "success": False,
            "error": f"Trajectory file does not exist: {trajectory_file}"
        }
    
    # 如果同时提供了结构File和轨迹File，use系统命令直接启动VMD
    if structure_file and trajectory_file:
        try:
            # 构建命令 - 在macOS上保证在后台运行
            if sys.platform == 'darwin':
                # 在macOS上useVMD启动脚本的完整路径
                struct_abs_path = os.path.abspath(structure_file)
                traj_abs_path = os.path.abspath(trajectory_file)
                
                # useos.system直接运行shell命令
                # 这更接近于在终端手动输入命令的行for
                cmd = f"vmd {struct_abs_path} {traj_abs_path} &"
                logger.info(f"useos.system直接启动VMD: {cmd}")
                
                # useos.system直接运行命令而不是通过asyncio
                os.system(cmd)
                
                # 由于我们use了os.system，我们无法获取进程ID
                # 但这种方式更接近于终端手动输入，更可能成功
                process_id = None
                
                # 不等待进程完成，让它在后台运行
                logger.info(f"VMD GUI已启动，加载结构File{structure_file}和轨迹File{trajectory_file}")
                
                return {
                    "success": True,
                    "pid": process_id,
                    "display": os.environ.get("DISPLAY", ":0"),
                    "message": "VMD图形界面已成功启动，并加载了结构和轨迹File",
                    "structure_file": structure_file,
                    "trajectory_file": trajectory_file
                }
            else:
                # 在其他系统上也use系统命令
                struct_abs_path = os.path.abspath(structure_file)
                traj_abs_path = os.path.abspath(trajectory_file)
                
                # 构建命令
                cmd = f"vmd {struct_abs_path} {traj_abs_path} &"
                logger.info(f"use系统命令启动VMD: {cmd}")
                
                # use系统命令运行VMD
                os.system(cmd)
                
                # 不等待进程完成，让它在后台运行
                logger.info(f"VMD GUI已启动，加载结构File{structure_file}和轨迹File{trajectory_file}")
                
                return {
                    "success": True,
                    "pid": None,
                    "display": os.environ.get("DISPLAY", ":0"),
                    "message": "VMD图形界面已成功启动，并加载了结构和轨迹File",
                    "structure_file": structure_file,
                    "trajectory_file": trajectory_file
                }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"启动VMD时发生异常: {error_msg}")
            return {
                "success": False,
                "error": f"启动VMD时发生异常: {error_msg}"
            }
    
    # 如果只提供了结构File或没有提供任何File，use原有方法
    return await service.vmd_manager.launch_gui(structure_file)

@mcp.tool("execute_vmd_script")
async def execute_vmd_script_tool(
    script: str, 
    process_id: Optional[int] = None, 
    structure_file: Optional[str] = None,
    generate_image: bool = False,
    image_file: Optional[str] = None
) -> Dict:
    """向VMD实例执行TCL脚本
    
    向已运行或新启动的VMD实例发送TCL脚本进行执行。可以选择性地加载分子结构并生成渲染图像。
    
    Args:
        script: TCL脚本内容
        process_id: 可选的VMD进程ID（若不提供则启动新实例）
        structure_file: 可选的分子结构File路径
        generate_image: 是否生成渲染图像
        image_file: 输出图像File路径（可选）
    
    Returns:
        Dict: 包含脚本执行结果和生成图像路径的字典
    """
    return await service.vmd_manager.execute_script(
        script, 
        process_id, 
        structure_file,
        generate_image,
        image_file
    )

@mcp.tool("close_vmd_instance")
async def close_vmd_instance_tool(process_id: int) -> Dict:
    """关闭指定的VMD实例
    
    根据进程ID关闭正在运行的VMD实例。
    
    Args:
        process_id: VMD实例的进程ID
    
    Returns:
        Dict: 操作结果
    """
    success = await service.vmd_manager.close_instance(process_id)
    return {
        "success": success,
        "message": f"VMD实例 {process_id} {'已关闭' if success else '关闭失败'}"
    }

@mcp.tool("list_vmd_instances")
async def list_vmd_instances_tool() -> Dict:
    """列出所有正在运行的VMD实例
    
    Returns:
        Dict: 包含所有VMD实例信息的字典
    """
    instances = service.vmd_manager.list_instances()
    return {
        "success": True,
        "instances": instances,
        "count": len(instances)
    }

# 分子动Force学模拟工具
@mcp.tool("prepare_simulation")
async def prepare_simulation_tool(
    workflow_id: str,
    structure_file: str,
    force_field: str = "amber99sb-ildn",
    simulation_params: Optional[Dict] = None
) -> Dict:
    """Prepare molecular dynamics simulation
    
    for分子动Force学模拟准备所有必要的输入File和parameter，包括系统构建、Energy minimization、平衡和Production simulation的配置。
    
    Args:
        workflow_id: Workflow ID
        structure_file: 结构File路径（PDBformat）
        force_field: Force field name（默认foramber99sb-ildn）
        simulation_params: 模拟parameter（可选）
    
    Returns:
        Dict: 包含准备好的输入File和命令的字典
    """
    # 检查是否有自定义工作流目录
    custom_dir = get_custom_workflow_directory(workflow_id)
    if custom_dir and workflow_id in workflow_dir_mapping:
        # 有Custom directory，Ensure directory permissions are correct
        ensure_workflow_directory_permissions(custom_dir)
        
        # 处理结构File路径 - 使其相对于工作流目录
        structure_path = Path(structure_file)
        if structure_path.is_absolute():
            # 将绝对路径转for相对路径
            try:
                rel_path = os.path.relpath(structure_path, custom_dir)
                structure_file = rel_path
            except ValueError:
                # 路径可能在不同驱动器上，无法获取相对路径，保持原样
                pass
        
        logger.info(f"use自定义工作流目录: {custom_dir}, 调整后的结构File路径: {structure_file}")
        
    # 调用服务方法准备模拟
    result = await service.prepare_simulation(
        workflow_id,
        structure_file,
        force_field,
        simulation_params,
        custom_workflow_dir=custom_dir  # 传递自定义Working directory
    )
    
    # 如果成功，确保工作流目录和所有子目录都有正确权限
    if result.get("success", False) and custom_dir:
        ensure_workflow_directory_permissions(custom_dir)
        
        # 再次检查关键目录是否存在并有正确权限
        for subdir in ["em", "nvt", "npt", "md"]:
            subdir_path = custom_dir / subdir
            if not subdir_path.exists():
                subdir_path.mkdir(parents=True, exist_ok=True)
            try:
                import stat
                os.chmod(subdir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except Exception as e:
                logger.warning(f"设置{subdir}目录权限失败: {e}")
    
    return result

# 配置工具
@mcp.tool("update_service_config")
async def update_config_tool(vmd_path: Optional[str] = None, gmx_path: Optional[str] = None) -> Dict:
    """更新服务配置
    
    更新VMD路径、GROMACS路径等服务配置。
    
    Args:
        vmd_path: VMD可执行File路径（可选）
        gmx_path: GROMACS可执行File路径（可选）
        
    Returns:
        Dict: 更新后的配置
    """
    return await update_config(vmd_path=vmd_path, gmx_path=gmx_path)

@mcp.tool("get_current_config")
async def get_config_tool() -> Dict:
    """获取当前服务配置
    
    Returns:
        Dict: 当前配置信息，包含VMD路径、结构搜索路径和GROMACS路径
    """
    return await get_config()

#====================
# 配置设置
#====================

@mcp.resource("gmx-vmd://config/update?vmd_path={vmd_path}&structure_search_paths={structure_search_paths}&gmx_path={gmx_path}")
async def update_config(vmd_path: Optional[str] = None, structure_search_paths: Optional[List[str]] = None, gmx_path: Optional[str] = None) -> Dict:
    """更新服务配置"""
    if vmd_path:
        vmd_config["vmd_path"] = vmd_path
        service.vmd_manager.vmd_path = vmd_path
        
    if structure_search_paths:
        # 清除现有搜索路径
        service.structure_search_paths.clear()
        # 添加新的搜索路径
        vmd_config["structure_search_paths"] = structure_search_paths
        for path in structure_search_paths:
            service.add_structure_search_path(path)
            
    if gmx_path:
        gmx_config["gmx_path"] = gmx_path
    
    # 保存配置到File
    try:
        config = {
            "vmd": vmd_config,
            "gmx": gmx_config
        }
        config_file = Path(os.getcwd()) / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"配置已保存到File: {config_file}")
    except Exception as e:
        logger.error(f"保存配置到File时出错: {e}")
    
    return {
        "success": True,
        "config": {**vmd_config, **gmx_config}
    }

@mcp.resource("gmx-vmd://config")
async def get_config() -> Dict:
    """获取当前配置"""
    return {**vmd_config, **gmx_config}

#====================
# 结构File搜索
#====================

@mcp.resource("gmx-vmd://structures/search?pattern={pattern}")
async def search_structures(pattern: str) -> Dict:
    """搜索结构File
    
    Args:
        pattern: 搜索模式，可以是File名、部分路径或结构名称
    """
    results = service.find_structure_files(pattern)
    return {
        "success": True,
        "count": len(results),
        "results": results
    }

#====================
# 工具定义
#====================

@mcp.tool("search_structure_files")
async def search_structures_tool(pattern: str) -> Dict:
    """搜索分子结构File
    
    根据File名、路径或结构名称搜索分子结构File。
    
    Args:
        pattern: 搜索模式，例如"1aki"或"protein"
        
    Returns:
        Dict: 包含搜索结果的字典
    """
    return await search_structures(pattern)

@mcp.tool("configure_search_paths")
async def configure_search_paths_tool(paths: List[str]) -> Dict:
    """配置结构File搜索路径
    
    设置在哪些目录下搜索分子结构File。
    
    Args:
        paths: 搜索路径列表
        
    Returns:
        Dict: 更新后的配置
    """
    return await update_config(structure_search_paths=paths)

@mcp.tool("modify_simulation_params")
async def modify_simulation_params_tool(
    workflow_id: str,
    instruction: str,
    stage: str = "all"  # 可以是 "minimization", "nvt", "npt", "production" 或 "all"
) -> Dict:
    """通过自然语言指令修改分子动Force学模拟parameter
    
    use自然语言描述修改模拟parameter的需求，系统会智能解析并应用到相应的mdpFile中。
    
    Args:
        workflow_id: Workflow ID
        instruction: 自然语言指令，例如"将temperature设置for310K"、"NVT平衡运行2ns"等
        stage: 要修改的模拟阶段，可选值for"minimization"、"nvt"、"npt"、"production"或"all"
        
    Returns:
        Dict: 包含修改结果的字典
    """
    # Get workflow information
    workflow = service.get_workflow(workflow_id)
    if not workflow:
        return {"success": False, "error": f"Workflow does not exist: {workflow_id}"}
        
    # 获取Working directory（useCustom directory）
    workflow_dir = get_custom_workflow_directory(workflow_id)
    if not workflow_dir:
        return {"success": False, "error": f"Cannot get workflow directory: {workflow_id}"}
    
    # 确保相关目录存在，并设置权限
    for subdir in ["em", "nvt", "npt", "md"]:
        subdir_path = os.path.join(workflow_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        try:
            import stat
            os.chmod(subdir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except Exception as e:
            logger.warning(f"设置{subdir}目录权限失败: {e}")
    
    # 定义要修改的mdpFile
    mdp_files = []
    if stage == "all" or stage == "minimization":
        mdp_files.append({"path": workflow_dir / "em" / "em.mdp", "stage": "Energy minimization"})
    if stage == "all" or stage == "nvt":
        mdp_files.append({"path": workflow_dir / "nvt" / "nvt.mdp", "stage": "NVT平衡"})
    if stage == "all" or stage == "npt":
        mdp_files.append({"path": workflow_dir / "npt" / "npt.mdp", "stage": "NPT平衡"})
    if stage == "all" or stage == "production":
        mdp_files.append({"path": workflow_dir / "md" / "md.mdp", "stage": "Production simulation"})
    
    # 检查mdpFile是否存在
    missing_files = []
    for mdp_file in mdp_files:
        if not os.path.exists(mdp_file["path"]):
            missing_files.append(f"{mdp_file['stage']}parameterFile({mdp_file['path']})")
    
    if missing_files:
        # 如果mdpFiledoes not exist，可能需要先运行准备模拟Step
        return {
            "success": False,
            "error": f"以下parameterFiledoes not exist: {', '.join(missing_files)}，请先运行'Prepare molecular dynamics simulation'工具"
        }
    
    # 解析自然语言指令并生成相应的修改
    modifications = parse_simulation_params_instruction(instruction)
    
    # 应用修改到mdpFile
    modified_files = []
    for mdp_file in mdp_files:
        # 读取原始mdpFile内容
        with open(mdp_file["path"], "r") as f:
            original_content = f.read()
        
        # 应用修改
        new_content = apply_mdp_modifications(original_content, modifications, mdp_file["stage"])
        
        # 如果内容有变化，写回File
        if new_content != original_content:
            with open(mdp_file["path"], "w") as f:
                f.write(new_content)
            modified_files.append(mdp_file["path"])
    
    # 返回修改结果
    if modified_files:
        logger.info(f"成功修改了以下File: {', '.join([str(path) for path in modified_files])}")
        logger.info(f"应用的修改: {modifications}")
        return {
            "success": True,
            "message": f"已成功修改以下parameterFile: {', '.join([str(path) for path in modified_files])}",
            "modifications": modifications,
            "affected_stages": [mdp_file["stage"] for mdp_file in mdp_files if mdp_file["path"] in modified_files]
        }
    else:
        logger.warning(f"未能应用任何修改，指令: '{instruction}'")
        logger.warning(f"解析结果: {modifications}")
        logger.warning(f"目标File: {[mdp_file['path'] for mdp_file in mdp_files]}")
        return {
            "success": False,
            "message": "未能应用任何修改，请检查您的指令是否有效",
            "instruction": instruction,
            "debug_info": {
                "parsed_modifications": modifications,
                "target_files": [str(mdp_file["path"]) for mdp_file in mdp_files],
                "file_exists": [os.path.exists(mdp_file["path"]) for mdp_file in mdp_files]
            }
        }

def parse_simulation_params_instruction(instruction: str) -> Dict:
    """解析自然语言指令，提取模拟parameter修改
    
    Args:
        instruction: 自然语言指令
        
    Returns:
        Dict: 包含parameter名称和值的字典
    """
    modifications = {}
    
    # 解析temperature设置
    if "temperature" in instruction or "temperature" in instruction.lower():
        # 匹配数字和单位K
        import re
        temp_match = re.search(r'(\d+(?:\.\d+)?)\s*[Kk]', instruction)
        if temp_match:
            modifications["temperature"] = float(temp_match.group(1))
    
    # 解析pressure设置
    if "pressure" in instruction or "压强" in instruction or "pressure" in instruction.lower():
        # 匹配数字和单位bar
        import re
        press_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:bar|巴)', instruction)
        if press_match:
            modifications["pressure"] = float(press_match.group(1))
    
    # 解析模拟时间设置
    if "时间" in instruction or "步数" in instruction or "步" in instruction or "time" in instruction.lower() or "step" in instruction.lower() or "运行" in instruction or "进行" in instruction:
        import re
        # 改进的正则表达式，更灵活地匹配数字和单位
        time_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:ns|纳秒|ps|皮秒)', instruction)
        if time_match:
            time_value = float(time_match.group(1))
            # 确定单位（默认forps）
            time_unit = "ps"
            if "ns" in instruction or "纳秒" in instruction:
                time_unit = "ns"
            
            # 转换forps
            if time_unit in ["ns", "纳秒"]:
                time_value *= 1000  # 转换forps
                
            modifications["simulation_time"] = time_value
            logger.info(f"从指令中提取的模拟时间: {time_value} ps (原始指令: '{instruction}')")
        else:
            logger.warning(f"无法从指令中提取模拟时间: '{instruction}'")
    
    # 解析Time step设置
    if "步长" in instruction or "time step" in instruction.lower() or "dt" in instruction.lower():
        import re
        # 匹配数字和单位fs或ps
        dt_match = re.search(r'(\d+(?:\.\d+)?)\s*(fs|飞秒|ps|皮秒)', instruction)
        if dt_match:
            dt_value = float(dt_match.group(1))
            dt_unit = dt_match.group(2)
            
            # 转换forps
            if dt_unit in ["fs", "飞秒"]:
                dt_value /= 1000  # 转换forps
                
            modifications["time_step"] = dt_value
    
    # 解析输出频率设置
    if "输出" in instruction or "轨迹" in instruction or "output" in instruction.lower() or "trajectory" in instruction.lower():
        import re
        # 匹配数字和单位ps或ns
        out_match = re.search(r'每\s*(\d+(?:\.\d+)?)\s*(ps|皮秒|ns|纳秒)', instruction)
        if out_match:
            out_value = float(out_match.group(1))
            out_unit = out_match.group(2)
            
            # 转换forps
            if out_unit in ["ns", "纳秒"]:
                out_value *= 1000  # 转换forps
                
            modifications["output_frequency"] = out_value
    
    # 解析约束设置
    if "约束" in instruction or "constraint" in instruction.lower():
        if "无约束" in instruction or "no constraint" in instruction.lower():
            modifications["constraints"] = "none"
        elif "氢键" in instruction or "h-bond" in instruction.lower():
            modifications["constraints"] = "h-bonds"
        elif "所有键" in instruction or "all bonds" in instruction.lower():
            modifications["constraints"] = "all-bonds"
    
    return modifications

def apply_mdp_modifications(mdp_content: str, modifications: Dict, stage: str) -> str:
    """应用parameter修改到mdpFile内容
    
    Args:
        mdp_content: 原始mdpFile内容
        modifications: 要应用的修改
        stage: 模拟阶段
        
    Returns:
        str: 修改后的mdpFile内容
    """
    import re
    lines = mdp_content.split('\n')
    modified_lines = []
    
    # 提取当前Time step
    dt_match = re.search(r'dt\s*=\s*(\d+(?:\.\d+)?)', mdp_content)
    current_dt = float(dt_match.group(1)) if dt_match else 0.002  # 默认值
    
    # 计算Steps per ps
    steps_per_ps = int(1.0 / current_dt)
    
    # 处理temperature修改
    if "temperature" in modifications and ("NVT" in stage or "NPT" in stage or "生产" in stage):
        temp_value = modifications["temperature"]
        # 修改ref_tparameter
        ref_t_pattern = re.compile(r'(ref_t\s*=\s*)(\d+(?:\.\d+)?)(.*)')
        temp_modified = False
        
        for line in lines:
            if ref_t_pattern.match(line):
                # 提取当前parameter值后面的部分（可能包含注释）
                match = ref_t_pattern.match(line)
                prefix = match.group(1)
                suffix = match.group(3)
                
                # 根据format可能是"ref_t = 300 300"这样的形式
                if " " in suffix.strip():
                    # 如果有多个temperature值，全部替换
                    groups = suffix.strip().split()
                    new_suffix = " ".join([str(temp_value)] * len(groups))
                    modified_lines.append(f"{prefix}{temp_value} {new_suffix}")
                else:
                    # 单个temperature值
                    modified_lines.append(f"{prefix}{temp_value}{suffix}")
                temp_modified = True
            else:
                modified_lines.append(line)
                
        # 如果没有找到并修改temperatureparameter，添加它
        if not temp_modified and ("NVT" in stage or "NPT" in stage or "生产" in stage):
            modified_lines.append(f"ref_t = {temp_value} {temp_value}  ; Modified by user instruction")
    else:
        modified_lines = lines
    
    # 处理pressure修改
    if "pressure" in modifications and ("NPT" in stage or "生产" in stage):
        press_value = modifications["pressure"]
        # 修改ref_pparameter
        ref_p_pattern = re.compile(r'(ref_p\s*=\s*)(\d+(?:\.\d+)?)(.*)')
        press_modified = False
        
        lines = modified_lines
        modified_lines = []
        
        for line in lines:
            if ref_p_pattern.match(line):
                # 提取当前parameter值后面的部分（可能包含注释）
                match = ref_p_pattern.match(line)
                prefix = match.group(1)
                suffix = match.group(3)
                
                modified_lines.append(f"{prefix}{press_value}{suffix}")
                press_modified = True
            else:
                modified_lines.append(line)
                
        # 如果没有找到并修改pressureparameter，添加它
        if not press_modified and ("NPT" in stage or "生产" in stage):
            modified_lines.append(f"ref_p = {press_value}  ; Modified by user instruction")
    
    # 处理模拟时间修改
    if "simulation_time" in modifications:
        time_value_ps = modifications["simulation_time"]  # 单位已转forps
        
        # 根据Time step计算步数
        nsteps = int(time_value_ps / current_dt)
        logger.info(f"基于Time step {current_dt} ps 计算的步数: {nsteps}")
        
        # 修改nstepsparameter
        nsteps_pattern = re.compile(r'(nsteps\s*=\s*)(\d+)(.*)')
        time_modified = False
        
        lines = modified_lines
        modified_lines = []
        
        for line in lines:
            if nsteps_pattern.match(line):
                # 提取当前parameter值后面的部分（可能包含注释）
                match = nsteps_pattern.match(line)
                prefix = match.group(1)
                old_value = match.group(2)
                suffix = match.group(3)
                
                logger.info(f"修改模拟步数: {old_value} -> {nsteps}")
                modified_lines.append(f"{prefix}{nsteps}{suffix}")
                time_modified = True
            else:
                modified_lines.append(line)
                
        # 如果没有找到并修改步数parameter，添加它
        if not time_modified:
            logger.info(f"未找到nstepsparameter，添加新行: nsteps = {nsteps}")
            modified_lines.append(f"nsteps = {nsteps}  ; Modified by user instruction")
    
    # 处理Time step修改
    if "time_step" in modifications and stage != "Energy minimization":  # Energy minimization不usedt
        dt_value = modifications["time_step"]  # 单位已转forps
        
        # 更新Time step
        dt_pattern = re.compile(r'(dt\s*=\s*)(\d+(?:\.\d+)?)(.*)')
        dt_modified = False
        
        lines = modified_lines
        modified_lines = []
        
        for line in lines:
            if dt_pattern.match(line):
                # 提取当前parameter值后面的部分（可能包含注释）
                match = dt_pattern.match(line)
                prefix = match.group(1)
                suffix = match.group(3)
                
                modified_lines.append(f"{prefix}{dt_value}{suffix}")
                dt_modified = True
            else:
                modified_lines.append(line)
                
        # 如果没有找到并修改Time stepparameter，添加它
        if not dt_modified and stage != "Energy minimization":
            modified_lines.append(f"dt = {dt_value}  ; Modified by user instruction")
            
        # 更新步数/ps
        steps_per_ps = int(1.0 / dt_value)
    
    # 处理输出频率修改
    if "output_frequency" in modifications:
        out_freq_ps = modifications["output_frequency"]  # 单位已转forps
        
        # 计算对应的步数
        out_steps = int(out_freq_ps * steps_per_ps)
        
        # 修改轨迹输出相关parameter
        output_params = {
            "nstxtcout": out_steps,    # Compressed trajectory输出频率
            "nstxout": out_steps * 10, # 完整轨迹输出频率
            "nstvout": out_steps * 10, # Velocity输出频率
            "nstfout": out_steps * 10, # Force输出频率
            "nstlog": out_steps,       # Log output频率
            "nstenergy": out_steps     # Energy output频率
        }
        
        lines = modified_lines
        modified_lines = []
        modified_params = set()
        
        # 修改已存在的parameter
        for line in lines:
            param_modified = False
            for param, value in output_params.items():
                param_pattern = re.compile(f'({param}\\s*=\\s*)(\\d+)(.*)')
                if param_pattern.match(line):
                    # 提取当前parameter值后面的部分（可能包含注释）
                    match = param_pattern.match(line)
                    prefix = match.group(1)
                    suffix = match.group(3)
                    
                    modified_lines.append(f"{prefix}{value}{suffix}")
                    modified_params.add(param)
                    param_modified = True
                    break
            
            if not param_modified:
                modified_lines.append(line)
        
        # 添加未找到的parameter
        for param, value in output_params.items():
            if param not in modified_params:
                modified_lines.append(f"{param} = {value}  ; Modified by user instruction")
    
    # 处理约束设置
    if "constraints" in modifications:
        constraint_value = modifications["constraints"]
        
        # 修改constraintsparameter
        constraints_pattern = re.compile(r'(constraints\s*=\s*)(\S+)(.*)')
        constraint_modified = False
        
        lines = modified_lines
        modified_lines = []
        
        for line in lines:
            if constraints_pattern.match(line):
                # 提取当前parameter值后面的部分（可能包含注释）
                match = constraints_pattern.match(line)
                prefix = match.group(1)
                suffix = match.group(3)
                
                modified_lines.append(f"{prefix}{constraint_value}{suffix}")
                constraint_modified = True
            else:
                modified_lines.append(line)
                
        # 如果没有找到并修改约束parameter，添加它
        if not constraint_modified:
            modified_lines.append(f"constraints = {constraint_value}  ; Modified by user instruction")
    
    # 检查是否真的修改了File内容
    result = '\n'.join(modified_lines)
    if "simulation_time" in modifications and 'nsteps' not in result:
        logger.warning(f"警告: 修改后的内容中没有找到nstepsparameter")
        
    return result

@mcp.tool("get_rmsd_example")
async def get_rmsd_analysis_example_tool(workflow_id: str) -> Dict:
    """获取RMSDanalysis的parameter示例
    
    根据指定的工作流生成RMSDanalysis所需的parameter示例。
    
    Args:
        workflow_id: Workflow ID
        
    Returns:
        Dict: 包含RMSDanalysisparameter示例的字典
    """
    # Get workflow directory（useCustom directory）
    workflow_dir = get_custom_workflow_directory(workflow_id)
    if not workflow_dir:
        return {
            "success": False,
            "error": f"Workflow directory does not exist: {workflow_id}"
        }
    
    # 尝试找到可用的轨迹File和结构File
    trajectory_files = []
    structure_files = []
    
    # 检查em目录
    em_dir = workflow_dir / "em"
    if os.path.exists(em_dir):
        if os.path.exists(em_dir / "em.xtc"):
            trajectory_files.append("em/em.xtc")
        if os.path.exists(em_dir / "em.trr"):
            trajectory_files.append("em/em.trr")
        if os.path.exists(em_dir / "em.gro"):
            structure_files.append("em/em.gro")
    
    # 检查nvt目录
    nvt_dir = workflow_dir / "nvt"
    if os.path.exists(nvt_dir):
        if os.path.exists(nvt_dir / "nvt.xtc"):
            trajectory_files.append("nvt/nvt.xtc")
        if os.path.exists(nvt_dir / "nvt.trr"):
            trajectory_files.append("nvt/nvt.trr")
        if os.path.exists(nvt_dir / "nvt.gro"):
            structure_files.append("nvt/nvt.gro")
    
    # 检查npt目录
    npt_dir = workflow_dir / "npt"
    if os.path.exists(npt_dir):
        if os.path.exists(npt_dir / "npt.xtc"):
            trajectory_files.append("npt/npt.xtc")
        if os.path.exists(npt_dir / "npt.trr"):
            trajectory_files.append("npt/npt.trr")
        if os.path.exists(npt_dir / "npt.gro"):
            structure_files.append("npt/npt.gro")
    
    # 检查md目录
    md_dir = workflow_dir / "md"
    if os.path.exists(md_dir):
        if os.path.exists(md_dir / "md.xtc"):
            trajectory_files.append("md/md.xtc")
        if os.path.exists(md_dir / "md.trr"):
            trajectory_files.append("md/md.trr")
        if os.path.exists(md_dir / "md.gro"):
            structure_files.append("md/md.gro")
    
    # 也检查根目录的结构File
    for file in os.listdir(workflow_dir):
        if file.endswith(".gro") and os.path.isfile(workflow_dir / file):
            structure_files.append(file)
    
    # 生成示例parameter
    example = {
        "analysis_type": "rmsd",
        "output_prefix": "rmsd_analysis",
        "selection": "protein",
        "begin_time": 0,
        "end_time": -1,
        "dt": 1
    }
    
    # 添加找到的File
    if trajectory_files:
        example["trajectory_file"] = trajectory_files[0]  # use第一个找到的轨迹File
    else:
        example["trajectory_file"] = "请提供轨迹File路径，例如：npt/npt.xtc"
        
    if structure_files:
        example["structure_file"] = structure_files[0]  # use第一个找到的结构File
    else:
        example["structure_file"] = "请提供结构File路径，例如：npt/npt.gro"
    
    return {
        "success": True,
        "message": "RMSDanalysisparameter示例",
        "example": example,
        "note": "请Note：analysis_type必须use小写'rmsd'，且必须提供structure_file和trajectory_fileparameter",
        "available_files": {
            "trajectory_files": trajectory_files,
            "structure_files": structure_files
        }
    }

# 导入VMD模块的部分
try:
    import vmd
    from vmd import molecule, display, animate, molrep, color, material, render, trans
    HAS_VMD_PYTHON = True
except ImportError:
    HAS_VMD_PYTHON = False
    logger.warning("未找到vmd-python模块，一些功能可能受限")

@mcp.tool("load_gromacs_trajectory")
async def load_gromacs_trajectory_tool(
    workflow_id: str,
    trajectory_file: str,
    structure_file: str,
    selection: str = "all",
    generate_image: bool = True,
    image_file: Optional[str] = None,
    representation: str = "动态新卡通"
) -> Dict:
    """加载并VisualizationGROMACSformat的分子模拟轨迹
    
    useVMD加载分子模拟轨迹并应用一些基本的Visualization设置。可以选择生成静态渲染图像。
    
    Args:
        workflow_id: Workflow ID
        trajectory_file: 轨迹File路径，相对于工作流程目录（如 md/md.xtc）或绝对路径
        structure_file: 结构File路径，相对于工作流程目录（如 md/md.gro）或绝对路径
        selection: VMD选择表达式（默认for"all"，选择所有原子）
        generate_image: 是否生成渲染图像
        image_file: 输出图像File名（可选），默认for"trajectory_view.png"
        representation: Visualization表现形式，可选值："动态新卡通"、"标准"、"卡通"、"VDW"、"Licorice"、"CPK"
    
    Returns:
        Dict: 包含加载结果和图像路径的字典
    """
    # 获取工作流程目录
    workflow_dir = get_custom_workflow_directory(workflow_id)
    if not workflow_dir:
        return {
            "success": False,
            "error": f"Workflow directory does not exist: {workflow_id}"
        }
    
    # 处理File路径（支持绝对路径和相对路径）
    logger.info(f"处理File路径: 轨迹={trajectory_file}, 结构={structure_file}, Working directory={workflow_dir}")
    
    # 构建完整的File路径（如果是相对路径，则添加Working directoryprefix）
    traj_path = trajectory_file if os.path.isabs(trajectory_file) else os.path.join(workflow_dir, trajectory_file)
    struct_path = structure_file if os.path.isabs(structure_file) else os.path.join(workflow_dir, structure_file)
    
    logger.info(f"最终处理后的File路径: 轨迹={traj_path}, 结构={struct_path}")
    
    # 检查File是否存在
    if not os.path.exists(traj_path):
        return {
            "success": False,
            "error": f"Trajectory file does not exist: {traj_path}"
        }
    
    if not os.path.exists(struct_path):
        return {
            "success": False,
            "error": f"Structure file does not exist: {struct_path}"
        }
    
    # 设置输出图像File
    if generate_image:
        if not image_file:
            image_file = "trajectory_view.png"
        # 处理图像File路径
        image_path = image_file if os.path.isabs(image_file) else os.path.join(workflow_dir, image_file)
        # 确保输出目录存在并有写权限
        image_dir = os.path.dirname(os.path.abspath(image_path))
        os.makedirs(image_dir, exist_ok=True)
        # 确保目录具有写权限
        try:
            import stat
            os.chmod(image_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except Exception as e:
            logger.warning(f"设置图像目录权限失败: {e}")
    else:
        image_path = None
    
    # 同时use两种方法加载轨迹：
    # 1. vmd-python方式（如果可用）
    # 2. 系统命令方式（作for备选）
    
    # 先尝试usevmd-python方式
    if HAS_VMD_PYTHON:
        try:
            logger.info("usevmd-python模块加载轨迹")
            # 创建新的Visualization会话
            molid = molecule.load('gro', struct_path)
            molecule.read(molid, 'xtc', traj_path)
            
            # 应用显示设置
            display.display(reset=True)
            display.projection("Orthographic")
            display.depthcue(False)
            display.rendermode("GLSL")
            color.color("Display", "Background", "white")
            
            # 设置动画模式 - 根据文档useanimate模块
            animate.activate(molid)
            animate.goto(0)  # 跳到第一帧
            animate.forward()  # 开始向前播放
            animate.once()  # 播放一次
            
            # 根据representation设置显示风格
            molrep.delrep(0, molid)  # 删除默认显示
            
            # 添加蛋白质表示
            if representation == "动态新卡通" or representation == "卡通":
                molrep.addrep(molid, 
                              selection=f"{selection} and (not water) and (not ions)",
                              style="NewCartoon",
                              color="Structure")
            elif representation == "标准":
                molrep.addrep(molid,
                              selection=f"{selection} and (not water) and (not ions)",
                              style="Lines",
                              color="Name")
            elif representation == "VDW":
                molrep.addrep(molid,
                              selection=f"{selection} and (not water) and (not ions)", 
                              style="VDW",
                              color="Name")
            elif representation == "Licorice":
                molrep.addrep(molid,
                              selection=f"{selection} and (not water) and (not ions)",
                              style="Licorice",
                              color="Name")
            elif representation == "CPK":
                molrep.addrep(molid,
                              selection=f"{selection} and (not water) and (not ions)", 
                              style="CPK",
                              color="Name")
            
            # 添加水分子点状表示
            molrep.addrep(molid, 
                          selection="water", 
                          style="Points", 
                          color="Name",
                          material="Transparent")
            
            # Add ionsVDW表示
            molrep.addrep(molid, 
                          selection="ions or name NA or name CL or name NA+ or name CL- or name CA or name MG or name ZN",
                          style="VDW",
                          color="Name")
            
            # 调整视角
            display.zoom(1.5)
            trans.rotate('x', -30)
            trans.rotate('y', 45)
            
            # 生成图像（如果需要）
            image_result = None
            if generate_image and image_path:
                render.render('snapshot', image_path)
                image_result = image_path
                logger.info(f"usevmd-python生成图像: {image_path}")
            
            # 成功usevmd-python加载轨迹
            return {
                "success": True,
                "message": "成功usevmd-python加载GROMACS轨迹",
                "vmd_process_id": None,  # vmd-python没有独立进程ID
                "trajectory_file": traj_path,
                "structure_file": struct_path,
                "image_path": image_result,
                "method": "vmd-python",
                "note": "VMD Python界面已加载，请在VMD窗口中查看"
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"usevmd-python加载轨迹时出错: {error_msg}")
            # 如果vmd-python方式失败，继续尝试系统命令方式
    
    # use系统命令方式作for备选
    try:
        logger.info("use系统命令方式加载VMD轨迹")
        
        # 生成一个非常简单的TCL脚本，只用于生成图像
        if generate_image and image_path:
            # 创建在公共临时目录中
            temp_dir = os.path.join(os.path.dirname(workflow_dir), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            try:
                import stat
                os.chmod(temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except Exception as e:
                logger.warning(f"设置临时目录权限失败: {e}")
            
            img_tcl_fd, img_tcl_path = tempfile.mkstemp(dir=temp_dir, suffix='.tcl')
            with os.fdopen(img_tcl_fd, 'w') as f:
                f.write(f"""
                # 加载结构和轨迹
                mol new "{struct_path}" type gro waitfor all
                mol addfile "{traj_path}" type xtc waitfor all
                
                # 调整视角
                display resetview
                color Display Background white
                scale by 1.5
                rotate x by -30
                rotate y by 45
                
                # 生成图像
                render snapshot "{image_path}" display %s
                exit
                """)
                
            # usetext模式运行VMD生成图像（不打开GUI）
            vmd_path = "/Applications/VMD.app/Contents/MacOS/startup.command" if sys.platform == 'darwin' else "vmd"
            img_cmd = f"{vmd_path} -dispdev text -e {img_tcl_path}"
            logger.info(f"执行VMD命令生成图像: {img_cmd}")
            
            # usesubprocess执行命令，等待完成
            import subprocess
            subprocess.run(img_cmd, shell=True, check=False)
            
            logger.info(f"图像生成完成: {image_path}")
        
        # use最直接的方法启动VMD查看器
        # 直接use终端命令，确保use绝对路径
        view_cmd = f"cd {os.path.dirname(struct_path)} && vmd {os.path.basename(struct_path)} {os.path.basename(traj_path)}"
        logger.info(f"启动VMD查看器命令: {view_cmd}")
        
        # 创建在公共临时目录中
        temp_dir = os.path.join(os.path.dirname(workflow_dir), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            import stat
            os.chmod(temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except Exception as e:
            logger.warning(f"设置临时目录权限失败: {e}")

        # 创建一个独立的脚本File来运行VMD
        script_fd, script_path = tempfile.mkstemp(dir=temp_dir, suffix='.sh')
        with os.fdopen(script_fd, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd {os.path.dirname(struct_path)}\n")
            # useVMD的完整路径
            if sys.platform == 'darwin':
                f.write(f"/Applications/VMD.app/Contents/MacOS/startup.command {os.path.basename(struct_path)} {os.path.basename(traj_path)}\n")
            else:
                f.write(f"vmd {os.path.basename(struct_path)} {os.path.basename(traj_path)}\n")
        
        # 使脚本可执行
        os.chmod(script_path, 0o755)
        
        # 在新终端窗口中运行脚本
        if sys.platform == 'darwin':
            # macOS上useopen命令在新终端中运行
            term_cmd = f"open -a Terminal {script_path}"
            subprocess.Popen(term_cmd, shell=True)
        else:
            # Linux上usex-terminal-emulator
            term_cmd = f"x-terminal-emulator -e '{script_path}'"
            subprocess.Popen(term_cmd, shell=True)
        
        return {
            "success": True,
            "message": "成功加载GROMACS轨迹到VMD",
            "vmd_process_id": None,
            "trajectory_file": traj_path,
            "structure_file": struct_path,
            "image_path": image_path if generate_image else None,
            "method": "terminal-launch",
            "note": "VMD已在新终端窗口中启动，请查看桌面"
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"加载GROMACS轨迹时发生异常: {error_msg}")
        
        return {
            "success": False,
            "error": f"加载GROMACS轨迹失败: {error_msg}"
        }

# Entry point for running the server
if __name__ == "__main__":
    mcp.run()