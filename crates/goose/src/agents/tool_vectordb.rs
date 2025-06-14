use anyhow::{Context, Result};
use arrow::array::{FixedSizeListBuilder, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use chrono::Local;
use etcetera::base_strategy::{BaseStrategy, Xdg};
use futures::TryStreamExt;
use lancedb::connect;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRecord {
    pub tool_name: String,
    pub description: String,
    pub schema: String,
    pub vector: Vec<f32>,
    pub extension_name: String,
}

pub struct ToolVectorDB {
    connection: Arc<RwLock<Connection>>,
    table_name: String,
}

impl ToolVectorDB {
    pub async fn new(table_name: Option<String>) -> Result<Self> {
        let db_path = Self::get_db_path()?;

        // Ensure the directory exists
        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("Failed to create database directory")?;
        }

        let connection = connect(db_path.to_str().unwrap())
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        let tool_db = Self {
            connection: Arc::new(RwLock::new(connection)),
            table_name: table_name.unwrap_or_else(|| "tools".to_string()),
        };

        // Initialize the table if it doesn't exist
        tool_db.init_table().await?;

        Ok(tool_db)
    }

    fn get_db_path() -> Result<PathBuf> {
        let data_dir = Xdg::new()
            .context("Failed to determine base strategy")?
            .data_dir();

        Ok(data_dir.join("goose").join("tool_db"))
    }

    async fn init_table(&self) -> Result<()> {
        let connection = self.connection.read().await;

        // Check if table exists
        let table_names = connection
            .table_names()
            .execute()
            .await
            .context("Failed to list tables")?;

        if !table_names.contains(&self.table_name) {
            // Create the table schema
            let schema = Arc::new(Schema::new(vec![
                Field::new("tool_name", DataType::Utf8, false),
                Field::new("description", DataType::Utf8, false),
                Field::new("schema", DataType::Utf8, false),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        1536, // OpenAI embedding dimension
                    ),
                    false,
                ),
                Field::new("extension_name", DataType::Utf8, false),
            ]));

            // Create empty table
            let tool_names = StringArray::from(vec![] as Vec<&str>);
            let descriptions = StringArray::from(vec![] as Vec<&str>);
            let schemas = StringArray::from(vec![] as Vec<&str>);
            let extension_names = StringArray::from(vec![] as Vec<&str>);

            // Create empty fixed size list array for vectors
            let mut vectors_builder =
                FixedSizeListBuilder::new(arrow::array::Float32Builder::new(), 1536);
            let vectors = vectors_builder.finish();

            let batch = arrow::record_batch::RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(tool_names),
                    Arc::new(descriptions),
                    Arc::new(schemas),
                    Arc::new(vectors),
                    Arc::new(extension_names),
                ],
            )
            .context("Failed to create record batch")?;
            // Create an empty table with the schema
            // LanceDB will create the table from the RecordBatch
            drop(connection);
            let connection = self.connection.write().await;

            // Use the RecordBatch directly
            let reader = arrow::record_batch::RecordBatchIterator::new(
                vec![Ok(batch)].into_iter(),
                schema.clone(),
            );

            connection
                .create_table(&self.table_name, Box::new(reader))
                .execute()
                .await
                .map_err(|e| {
                    anyhow::anyhow!("Failed to create tools table '{}': {}", self.table_name, e)
                })?;
        }

        Ok(())
    }

    #[cfg(test)]
    pub async fn clear_tools(&self) -> Result<()> {
        let connection = self.connection.write().await;

        // Try to open the table first
        match connection.open_table(&self.table_name).execute().await {
            Ok(table) => {
                // Delete all records instead of dropping the table
                table
                    .delete("1=1") // This will match all records
                    .await
                    .context("Failed to delete all records")?;
            }
            Err(_) => {
                // If table doesn't exist, that's fine - we'll create it
            }
        }

        drop(connection);

        // Ensure table exists with correct schema
        self.init_table().await?;

        Ok(())
    }

    pub async fn index_tools(&self, tools: Vec<ToolRecord>) -> Result<()> {
        if tools.is_empty() {
            return Ok(());
        }

        let tool_names: Vec<&str> = tools.iter().map(|t| t.tool_name.as_str()).collect();
        let descriptions: Vec<&str> = tools.iter().map(|t| t.description.as_str()).collect();
        let schemas: Vec<&str> = tools.iter().map(|t| t.schema.as_str()).collect();
        let extension_names: Vec<&str> = tools.iter().map(|t| t.extension_name.as_str()).collect();

        let vectors_data: Vec<Option<Vec<Option<f32>>>> = tools
            .iter()
            .map(|t| Some(t.vector.iter().map(|&v| Some(v)).collect()))
            .collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("tool_name", DataType::Utf8, false),
            Field::new("description", DataType::Utf8, false),
            Field::new("schema", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    1536,
                ),
                false,
            ),
            Field::new("extension_name", DataType::Utf8, false),
        ]));

        let tool_names_array = StringArray::from(tool_names);
        let descriptions_array = StringArray::from(descriptions);
        let schemas_array = StringArray::from(schemas);
        let extension_names_array = StringArray::from(extension_names);
        // Build vectors array
        let mut vectors_builder =
            FixedSizeListBuilder::new(arrow::array::Float32Builder::new(), 1536);
        for vector_opt in vectors_data {
            if let Some(vector) = vector_opt {
                let values = vectors_builder.values();
                for val_opt in vector {
                    if let Some(val) = val_opt {
                        values.append_value(val);
                    } else {
                        values.append_null();
                    }
                }
                vectors_builder.append(true);
            } else {
                vectors_builder.append(false);
            }
        }
        let vectors_array = vectors_builder.finish();

        let batch = arrow::record_batch::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(tool_names_array),
                Arc::new(descriptions_array),
                Arc::new(schemas_array),
                Arc::new(vectors_array),
                Arc::new(extension_names_array),
            ],
        )
        .context("Failed to create record batch")?;

        let connection = self.connection.read().await;
        let table = connection
            .open_table(&self.table_name)
            .execute()
            .await
            .context("Failed to open tools table")?;

        // Add batch to table using RecordBatchIterator
        let reader = arrow::record_batch::RecordBatchIterator::new(
            vec![Ok(batch)].into_iter(),
            schema.clone(),
        );

        table
            .add(Box::new(reader))
            .execute()
            .await
            .context("Failed to add tools to table")?;

        Ok(())
    }

    pub async fn search_tools(
        &self,
        query_vector: Vec<f32>,
        k: usize,
        extension_name: Option<&str>,
    ) -> Result<Vec<ToolRecord>> {
        let connection = self.connection.read().await;

        let table = connection
            .open_table(&self.table_name)
            .execute()
            .await
            .context("Failed to open tools table")?;

        let search = table
            .vector_search(query_vector)
            .context("Failed to create vector search")?;

        let results = search
            .limit(k)
            .execute()
            .await
            .context("Failed to execute vector search")?;

        let batches: Vec<_> = results.try_collect().await?;

        let mut tools = Vec::new();
        for batch in batches {
            let tool_names = batch
                .column_by_name("tool_name")
                .context("Missing tool_name column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid tool_name column type")?;

            let descriptions = batch
                .column_by_name("description")
                .context("Missing description column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid description column type")?;

            let schemas = batch
                .column_by_name("schema")
                .context("Missing schema column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid schema column type")?;

            let extension_names = batch
                .column_by_name("extension_name")
                .context("Missing extension_name column")?
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Invalid extension_name column type")?;

            // Get the distance scores
            let distances = batch
                .column_by_name("_distance")
                .context("Missing _distance column")?
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .context("Invalid _distance column type")?;

            for i in 0..batch.num_rows() {
                let tool_name = tool_names.value(i).to_string();
                let _distance = distances.value(i);
                let ext_name = extension_names.value(i).to_string();

                // Filter by extension name if provided
                if let Some(filter_ext) = extension_name {
                    if ext_name != filter_ext {
                        continue;
                    }
                }

                tools.push(ToolRecord {
                    tool_name,
                    description: descriptions.value(i).to_string(),
                    schema: schemas.value(i).to_string(),
                    vector: vec![], // We don't need to return the vector
                    extension_name: ext_name,
                });
            }
        }
        Ok(tools)
    }

    pub async fn remove_tool(&self, tool_name: &str) -> Result<()> {
        let connection = self.connection.read().await;

        let table = connection
            .open_table(&self.table_name)
            .execute()
            .await
            .context("Failed to open tools table")?;

        // Delete records matching the tool name
        table
            .delete(&format!("tool_name = '{}'", tool_name))
            .await
            .context("Failed to delete tool")?;

        Ok(())
    }
}

pub fn generate_table_id() -> String {
    Local::now().format("%Y%m%d_%H%M%S").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_vectordb_creation() {
        let db = ToolVectorDB::new(Some("test_tools_vectordb_creation".to_string()))
            .await
            .unwrap();
        db.clear_tools().await.unwrap();
        assert_eq!(db.table_name, "test_tools_vectordb_creation");
    }

    #[tokio::test]
    async fn test_tool_vectordb_operations() -> Result<()> {
        // Create a new database instance with a unique table name
        let db = ToolVectorDB::new(Some("test_tool_vectordb_operations".to_string())).await?;

        // Clear any existing tools
        db.clear_tools().await?;

        // Create test tool records
        let test_tools = vec![
            ToolRecord {
                tool_name: "test_tool_1".to_string(),
                description: "A test tool for reading files".to_string(),
                schema: r#"{"type": "object", "properties": {"path": {"type": "string"}}}"#
                    .to_string(),
                vector: vec![0.1; 1536], // Mock embedding vector
                extension_name: "test_extension".to_string(),
            },
            ToolRecord {
                tool_name: "test_tool_2".to_string(),
                description: "A test tool for writing files".to_string(),
                schema: r#"{"type": "object", "properties": {"path": {"type": "string"}}}"#
                    .to_string(),
                vector: vec![0.2; 1536], // Different mock embedding vector
                extension_name: "test_extension".to_string(),
            },
        ];

        // Index the test tools
        db.index_tools(test_tools).await?;

        // Search for tools using a query vector similar to test_tool_1
        let query_vector = vec![0.1; 1536];
        let results = db.search_tools(query_vector.clone(), 2, None).await?;

        // Verify results
        assert_eq!(results.len(), 2, "Should find both tools");
        assert_eq!(
            results[0].tool_name, "test_tool_1",
            "First result should be test_tool_1"
        );
        assert_eq!(
            results[1].tool_name, "test_tool_2",
            "Second result should be test_tool_2"
        );

        // Test filtering by extension name
        let results = db
            .search_tools(query_vector.clone(), 2, Some("test_extension"))
            .await?;
        assert_eq!(
            results.len(),
            2,
            "Should find both tools with test_extension"
        );

        let results = db
            .search_tools(query_vector.clone(), 2, Some("nonexistent_extension"))
            .await?;
        assert_eq!(
            results.len(),
            0,
            "Should find no tools with nonexistent_extension"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_db() -> Result<()> {
        // Create a new database instance with a unique table name
        let db = ToolVectorDB::new(Some("test_empty_db".to_string())).await?;

        // Clear any existing tools
        db.clear_tools().await?;

        // Search in empty database
        let query_vector = vec![0.1; 1536];
        let results = db.search_tools(query_vector, 2, None).await?;

        // Verify no results returned
        assert_eq!(results.len(), 0, "Empty database should return no results");

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_deletion() -> Result<()> {
        // Create a new database instance with a unique table name
        let db = ToolVectorDB::new(Some("test_tool_deletion".to_string())).await?;

        // Clear any existing tools
        db.clear_tools().await?;

        // Create and index a test tool
        let test_tool = ToolRecord {
            tool_name: "test_tool_to_delete".to_string(),
            description: "A test tool that will be deleted".to_string(),
            schema: r#"{"type": "object", "properties": {"path": {"type": "string"}}}"#.to_string(),
            vector: vec![0.1; 1536],
            extension_name: "test_extension".to_string(),
        };

        db.index_tools(vec![test_tool]).await?;

        // Verify tool exists
        let query_vector = vec![0.1; 1536];
        let results = db.search_tools(query_vector.clone(), 1, None).await?;
        assert_eq!(results.len(), 1, "Tool should exist before deletion");

        // Delete the tool
        db.remove_tool("test_tool_to_delete").await?;

        // Verify tool is gone
        let results = db.search_tools(query_vector.clone(), 1, None).await?;
        assert_eq!(results.len(), 0, "Tool should be deleted");

        Ok(())
    }
}
