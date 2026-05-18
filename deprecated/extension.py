# implement this code to pdf_graphrag.py
# its role is to extense graphrag with docker management for neo4j


class PDFGraphRAG:
    
    # Add this as a class variable to track container state
    CONTAINER_NAME = "llm-knowledge-graph-neo4j"
    
    def __init__(self, vector_store_chunk_name: str, vector_store_nodes_name: str, 
                 vector_store_relationships_name: str, neo4j_uri: str, neo4j_user: str, 
                 neo4j_password: str, openai_api_key: str, google_api_key: str = None,
                 use_docker: bool = True, auto_cleanup: bool = False):
        """
        Initialize PDFGraphRAG with optional Docker Neo4j management.
        
        Args:
            use_docker: If True, manage Neo4j via Docker
            auto_cleanup: If True, stop Docker container on exit
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.use_docker = use_docker
        self.auto_cleanup = auto_cleanup
        
        if use_docker:
            self._setup_neo4j_docker(neo4j_uri, neo4j_user, neo4j_password)
        
        # Initialize graph connection
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False
        )
        
        # Rest of your existing __init__ code continues here...
        try:
            self.vector_store_chunk = Neo4jVector.from_existing_index(
                self.embeddings,
                url=neo4j_uri,
                username=neo4j_user,
                password=neo4j_password,
                index_name=vector_store_chunk_name,
            )
            # ... etc
        except Exception as e:
            print(f"Error initializing vector stores: {e}")
            raise e
        
        # ... rest of init code ...
        
        # Register cleanup on exit if auto_cleanup is enabled
        if self.auto_cleanup:
            import atexit
            atexit.register(self.close_docker)
    
    
    def _setup_neo4j_docker(self, uri: str, user: str, password: str):
        """Setup Neo4j Docker container - create if not exists, start if stopped"""
        port = uri.split(":")[-1]  # Extract port from bolt://localhost:7687
        
        try:
            # Check if Docker is installed
            subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5,
                check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ERROR: Docker is not installed or not in PATH")
            print("Install Docker Desktop from: https://www.docker.com/products/docker-desktop")
            raise
        
        try:
            # Check if container exists and get its state
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.CONTAINER_NAME}", "--format", "{{.State}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            container_state = result.stdout.strip()
            
            if not container_state:
                # Container doesn't exist, create it
                print(f"🔨 Creating Neo4j Docker container '{self.CONTAINER_NAME}'...")
                subprocess.run([
                    "docker", "run", "-d",
                    "--name", self.CONTAINER_NAME,
                    "-p", f"{port}:7687",
                    "-p", "7474:7474",
                    "-e", f"NEO4J_AUTH={user}/{password}",
                    "-e", "NEO4J_dbms_memory_pagecache_size=256M",
                    "-e", "NEO4J_dbms_memory_heap_initial__size=256M",
                    "-e", "NEO4J_dbms_memory_heap_max__size=512M",
                    "-e", "NEO4J_ACCEPT_LICENSE_AGREEMENT=yes",
                    "neo4j:latest"
                ], timeout=60, check=True)
                
                print(f"✓ Container created. Waiting for Neo4j to start...")
                self._wait_for_neo4j(uri, user, password)
                
            elif container_state == "exited":
                # Container exists but is stopped, start it
                print(f"🔄 Starting existing Neo4j container '{self.CONTAINER_NAME}'...")
                subprocess.run(["docker", "start", self.CONTAINER_NAME], timeout=10, check=True)
                print(f"✓ Container started. Waiting for Neo4j to be ready...")
                self._wait_for_neo4j(uri, user, password)
            
            else:
                # Container is already running
                print(f"✓ Neo4j container '{self.CONTAINER_NAME}' is already running")
                # Still verify it's responsive
                self._wait_for_neo4j(uri, user, password, max_retries=3)
                
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Docker command failed: {e}")
            raise
        except Exception as e:
            print(f"ERROR setting up Neo4j Docker container: {e}")
            raise
    
    
    def _wait_for_neo4j(self, uri: str, user: str, password: str, max_retries: int = 30):
        """Wait for Neo4j to be ready by attempting connections"""
        retry_count = 0
        while retry_count < max_retries:
            try:
                driver = GraphDatabase.driver(uri, auth=(user, password))
                with driver.session() as session:
                    session.run("RETURN 1")
                driver.close()
                print("✓ Neo4j is ready and responding!")
                return
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2
                    print(f"⏳ Waiting for Neo4j... ({retry_count}/{max_retries}) [waiting {wait_time}s]")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Neo4j failed to start after {max_retries * 2} seconds")
                    raise
    
    
    def close_docker(self, stop: bool = True):
        """
        Close/quit the Docker container.
        
        Args:
            stop: If True, stop the container. If False, only disconnect from it.
        """
        if not self.use_docker:
            print("Docker management not enabled")
            return
        
        try:
            # Close graph connection first
            if hasattr(self, 'graph'):
                self.graph.close()
                print("✓ Graph connection closed")
            
            if stop:
                print(f"🛑 Stopping Neo4j container '{self.CONTAINER_NAME}'...")
                result = subprocess.run(
                    ["docker", "stop", self.CONTAINER_NAME],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"✓ Container stopped successfully")
                else:
                    print(f"⚠ Warning: Could not stop container: {result.stderr}")
            else:
                print(f"✓ Disconnected from Neo4j (container still running)")
                
        except subprocess.CalledProcessError as e:
            print(f"ERROR stopping container: {e}")
        except Exception as e:
            print(f"ERROR closing Docker: {e}")
    
    
    def remove_docker_container(self):
        """
        Completely remove the Docker container (dangerous - deletes data).
        Use this to reset/clean up.
        """
        if not self.use_docker:
            print("Docker management not enabled")
            return
        
        try:
            # First stop the container
            self.close_docker(stop=True)
            
            # Then remove it
            print(f"🗑️  Removing Neo4j container '{self.CONTAINER_NAME}'...")
            result = subprocess.run(
                ["docker", "rm", self.CONTAINER_NAME],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✓ Container removed successfully. Data has been deleted.")
            else:
                print(f"⚠ Warning: Could not remove container: {result.stderr}")
                
        except Exception as e:
            print(f"ERROR removing container: {e}")
